use std::{
    marker::PhantomData,
    sync::atomic::{AtomicI64, AtomicU64, Ordering},
    thread,
};

use dashmap::DashMap;
use float_ord::FloatOrd;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::{Action, MCTSInfo, Player, State, StateNode, StateNodeInfo, MCTS};

const MAX_ROLLOUT_DEPTH: usize = 200;

impl<A: Action> StateNode<A> {
    fn new(available_actions: Vec<A>, player_len: usize) -> Self {
        let mut total_reward = Vec::with_capacity(player_len);

        for _ in 0..player_len {
            total_reward.push(AtomicI64::new(0));
        }

        Self {
            available_actions,
            total_reward,
            total_visit: AtomicU64::new(0),
        }
    }
}

impl MCTSInfo {
    fn new() -> Self {
        Self {
            total_node: AtomicU64::new(0),
            ignored_rollout: AtomicU64::new(0),
        }
    }
}

impl<P: Player, A: Action, S: State<P, A>> MCTS<P, A, S> {
    pub fn new() -> Self {
        Self {
            state_map: DashMap::new(),
            info: MCTSInfo::new(),
            _marker: PhantomData,
        }
    }

    pub fn new_with_capacity(capacity: usize) -> Self {
        Self {
            state_map: DashMap::with_capacity(capacity),
            info: MCTSInfo::new(),
            _marker: PhantomData,
        }
    }

    fn select_by_uct(&self, state: S, node: &StateNode<A>) -> Option<A> {
        let now_pid = state.current_player().id();

        let ln_parent_total_visit = ((node.total_visit.load(Ordering::Relaxed) + 1) as f64).ln();

        let uct = |action: &A| {
            let child_node = self
                .state_map
                .get(&state.clone().apply_action((*action).clone()));

            let child_uct_factor = child_node.map_or(f64::INFINITY, |node| {
                let reward = node.total_reward[now_pid].load(Ordering::Relaxed) as f64;
                let visit = node.total_visit.load(Ordering::Relaxed) as f64;

                if visit == 0. {
                    f64::INFINITY
                } else {
                    // refer to https://gusals1620.tistory.com/5
                    const EXPLORATION_CONSTANT: f64 = 1.4142;

                    (reward / visit)
                        + EXPLORATION_CONSTANT * 2. * ((2. * ln_parent_total_visit / visit).sqrt())
                }
            });

            FloatOrd(child_uct_factor)
        };

        let mut choice = None;
        let mut same_choices_num = 0;
        let mut max_score = FloatOrd(f64::NEG_INFINITY);

        let mut rng: XorShiftRng = SeedableRng::from_entropy();

        for action in node.available_actions.iter() {
            let score = uct(action);

            if score > max_score {
                choice = Some(action.clone());
                same_choices_num = 1;
                max_score = score;
            } else if score == max_score {
                same_choices_num += 1;

                if rng.gen_ratio(1, same_choices_num) {
                    choice = Some(action.clone());
                }
            }
        }

        choice
    }

    fn backpropagate(&self, path: Vec<S>, rewards: Vec<i64>) {
        for state in path.iter().rev() {
            let node = self.state_map.get(state).unwrap();

            node.total_visit.fetch_add(1, Ordering::Relaxed);
            for (i, reward) in rewards.iter().enumerate() {
                node.total_reward[i].fetch_add(*reward, Ordering::Relaxed);
            }
        }
    }

    pub fn rollout_from(&self, mut state: S) {
        let mut path = Vec::new();

        loop {
            path.push(state.clone());
            if path.len() > MAX_ROLLOUT_DEPTH {
                self.info.ignored_rollout.fetch_add(1, Ordering::Relaxed);
                return;
            }

            let node = self.state_map.get(&state).unwrap_or_else(|| {
                let result = self
                    .state_map
                    .entry(state.clone())
                    .or_insert_with(|| {
                        self.info.total_node.fetch_add(1, Ordering::Relaxed);
                        StateNode::new(state.available_actions(), state.player_len())
                    })
                    .downgrade();
                result
            });

            let next_action = self.select_by_uct(state.clone(), &node).unwrap();
            state.do_action(next_action);

            if let Some(rewards) = state.game_over() {
                self.backpropagate(path, rewards);
                return;
            }
        }
    }

    pub fn rollout_iter_from(&self, state: S, iter: u64) {
        for _ in 0..iter {
            self.rollout_from(state.clone());
        }
    }

    pub fn rollout_parallel_from(&self, state: S, thread_num: usize, total_iter: u64) {
        assert!(
            total_iter < (1 << 60),
            "There should be margin of iter num."
        );
        let counter = AtomicU64::new(0);

        thread::scope(|scope| {
            for _ in 0..thread_num {
                scope.spawn(|| loop {
                    let count = counter.fetch_add(1, Ordering::Relaxed);

                    if count >= total_iter {
                        break;
                    }

                    self.rollout_from(state.clone());
                });
            }
        });
    }

    pub fn node_info_from(&self, state: &S) -> Option<StateNodeInfo> {
        self.state_map.get(state).map(|node| StateNodeInfo {
            total_reward: node
                .total_reward
                .iter()
                .map(|reward| reward.load(Ordering::Relaxed))
                .collect(),
            total_visit: node.total_visit.load(Ordering::Relaxed),
        })
    }

    pub fn best_actions_from(&self, state: &S, max_num: usize) -> Vec<A> {
        let now_pid = state.current_player().id();

        let node = self.state_map.get(state);

        if node.is_none() {
            return Vec::new();
        }

        let mut actions = node.unwrap().available_actions.clone();
        actions.sort_by_cached_key(|action| {
            let child_node = self
                .state_map
                .get(&state.clone().apply_action((*action).clone()));

            if let Some(node) = child_node {
                let reward = node.total_reward[now_pid].load(Ordering::Relaxed) as f64;
                let visit = node.total_visit.load(Ordering::Relaxed) as f64;

                if visit > 0. {
                    return FloatOrd(-1. * reward / visit);
                }
            }

            FloatOrd(-1. * f64::NEG_INFINITY)
        });

        if actions.len() <= max_num {
            actions
        } else {
            (&actions[0..max_num]).to_vec()
        }
    }

    pub fn diagnose(&self) -> String {
        self.info.to_string()
    }
}
