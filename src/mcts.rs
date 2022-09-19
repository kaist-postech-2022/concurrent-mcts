use std::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    thread,
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use float_ord::FloatOrd;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::*;

impl<A: Action> StateNode<A> {
    fn new(available_actions: Vec<A>, player_len: usize) -> Self {
        let mut total_reward = Vec::with_capacity(player_len);

        for _ in 0..player_len {
            total_reward.push(AtomicF64::new(0.));
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
            total_rollout: AtomicU64::new(0),
            partial_rewards: AtomicU64::new(0),
            ignored_max_depth_rollout: AtomicU64::new(0),
            ignored_try_cycle_by_uct: AtomicU64::new(0),
            ignored_cycle_rollout: AtomicU64::new(0),
        }
    }
}

impl<P: Player, A: Action, S: State<P, A>> MCTS<P, A, S> {
    pub fn new(option: MCTSOption) -> Self {
        Self {
            state_map: DashMap::new(),
            info: MCTSInfo::new(),
            option,
            _marker: PhantomData,
        }
    }

    pub fn new_with_capacity(option: MCTSOption, capacity: usize) -> Self {
        Self {
            state_map: DashMap::with_capacity(capacity),
            info: MCTSInfo::new(),
            option,
            _marker: PhantomData,
        }
    }

    pub fn new_with_max_depth_and_capacity(option: MCTSOption, capacity: usize) -> Self {
        Self {
            state_map: DashMap::with_capacity(capacity),
            info: MCTSInfo::new(),
            option,
            _marker: PhantomData,
        }
    }

    /// Select next action with UCT policy
    ///
    /// The exploration constant is sqrt(2) according to AlphaGo.
    fn select_by_uct(&self, state: S, node: &StateNode<A>, ignore_states: &Vec<S>) -> Option<A> {
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
                        + action.choose_weight()
                            * EXPLORATION_CONSTANT
                            * 2.
                            * ((2. * ln_parent_total_visit / visit).sqrt())
                }
            });

            FloatOrd(child_uct_factor)
        };

        let mut choice = None;
        let mut same_choices_num = 0;
        let mut max_score = FloatOrd(f64::NEG_INFINITY);

        let mut rng: XorShiftRng = SeedableRng::from_entropy();

        for action in node.available_actions.iter() {
            if self.option.cycle_policy == CyclePolicy::Ignore
                && ignore_states.contains(&state.clone().apply_action(action.clone()))
            {
                self.info
                    .ignored_try_cycle_by_uct
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

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

    fn backpropagate(&self, path: Vec<S>, rewards: Vec<f64>) {
        for state in path.iter().rev() {
            let node = self.state_map.get(state).unwrap();

            node.total_visit.fetch_add(1, Ordering::Relaxed);
            for (i, reward) in rewards.iter().enumerate() {
                node.total_reward[i].fetch_add(*reward, Ordering::Relaxed);
            }
        }

        self.info.total_rollout.fetch_add(1, Ordering::Relaxed);
    }

    /// Rollout one time and backpropagate the win result except over-depth
    ///
    /// The default policy of cycle on tree is ignore. If you want to do other policy, please re-implement.
    pub fn rollout_from(&self, mut state: S) {
        let mut path = Vec::new();

        loop {
            if path.len() > self.option.max_depth {
                if self.option.cycle_policy == CyclePolicy::PartialReward {
                    self.info.partial_rewards.fetch_add(1, Ordering::Relaxed);
                    self.backpropagate(path, state.partial_rewards());
                } else {
                    self.info
                        .ignored_max_depth_rollout
                        .fetch_add(1, Ordering::Relaxed);
                }

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

            path.push(state.clone());

            if let Some(rewards) = state.game_over() {
                self.backpropagate(path, rewards);
                return;
            }

            if let Some(next_action) = self.select_by_uct(state.clone(), &node, &path) {
                state.do_action(next_action);

                if path.contains(&state) && self.option.cycle_policy == CyclePolicy::PartialReward {
                    self.info.partial_rewards.fetch_add(1, Ordering::Relaxed);
                    self.backpropagate(path, state.partial_rewards());
                    return;
                }
            } else {
                self.info
                    .ignored_cycle_rollout
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    pub fn rollout_iter_from(&self, state: S, iter: u64) {
        for _ in 0..iter {
            self.rollout_from(state.clone());
        }
    }

    pub fn rollout_parallel_iter_from(&self, state: S, thread_num: usize, total_iter: u64) {
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

    /// Run rollout with parallel during almost given duration
    ///
    /// Pay attention to that it does not obey strictly the given duration. If you really want to obey strictly it, implement unsafe version that kills threads after the given duration
    /// or run this function with async and do what you want afther the given duration. It is available since it is thread-safe.
    pub fn rollout_parallel_until_from(&self, state: S, thread_num: usize, until: Duration) {
        let flag = AtomicBool::new(true);

        thread::scope(|scope| {
            let start = SystemTime::now();

            let mut test_threads = Vec::new();

            for _ in 0..thread_num {
                let thread = scope.spawn(|| {
                    for _ in 0..5 {
                        self.rollout_from(state.clone());
                    }
                });

                test_threads.push(thread);
            }

            for t in test_threads {
                t.join().unwrap();
            }

            let mut threads = Vec::with_capacity(thread_num);

            let temp_time = start.elapsed().unwrap();
            let avg_par_time = temp_time / 5;

            for _ in 0..thread_num {
                let thread = scope.spawn(|| {
                    while flag.load(Ordering::Relaxed) {
                        self.rollout_from(state.clone());
                    }
                });

                threads.push(thread);
            }

            thread::sleep(until - temp_time - 3 * avg_par_time);

            flag.store(false, Ordering::SeqCst);

            for t in threads {
                t.join().unwrap();
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

    pub fn best_actions_from(&self, state: &S, max_num: usize) -> Vec<(A, FloatOrd<f64>)> {
        let now_pid = state.current_player().id();

        let node = self.state_map.get(state);

        if node.is_none() {
            return Vec::new();
        }

        let actions = node.unwrap().available_actions.clone();
        let mut actions = actions
            .into_iter()
            .map(|action| {
                let child_node = self
                    .state_map
                    .get(&state.clone().apply_action(action.clone()));

                if let Some(node) = child_node {
                    let reward = node.total_reward[now_pid].load(Ordering::Relaxed) as f64;
                    let visit = node.total_visit.load(Ordering::Relaxed) as f64;

                    if visit > 0. {
                        return (action, FloatOrd(-1. * reward / visit));
                    }
                }

                (action, FloatOrd(-1. * f64::NEG_INFINITY))
            })
            .collect::<Vec<_>>();

        actions.sort_by_key(|action| action.1);

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
