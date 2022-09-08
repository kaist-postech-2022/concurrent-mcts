pub mod mcts;

use std::{
    fmt,
    hash::Hash,
    marker::PhantomData,
    sync::atomic::{AtomicI64, AtomicU64, Ordering},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

pub trait Action: Clone + Send + Sync {}

pub trait Player: Clone + Send + Sync {
    fn id(&self) -> usize;
}

pub trait State<P: Player, A: Action>: Clone + Hash + PartialEq + Eq + Send + Sync {
    fn player_len(&self) -> usize;

    fn current_player(&self) -> P;

    fn available_actions(&self) -> Vec<A>;

    fn do_action(&mut self, action: A);

    fn apply_action(mut self, action: A) -> Self {
        self.do_action(action);
        self
    }

    // if the state is game over, return the reward of each players by id.
    fn game_over(&self) -> Option<Vec<i64>>;
}

#[derive(Serialize, Deserialize)]
struct StateNode<A: Action> {
    available_actions: Vec<A>,
    total_reward: Vec<AtomicI64>,
    total_visit: AtomicU64,
}

/// The node info for public
///
/// This can generated by `self.node_info_from`
#[derive(Debug)]
pub struct StateNodeInfo {
    total_reward: Vec<i64>,
    total_visit: u64,
}

#[derive(Serialize, Deserialize)]
struct MCTSInfo {
    total_node: AtomicU64,
}

impl fmt::Display for MCTSInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCTSInfo")
            .field(
                "total_node",
                &thousands_separate(self.total_node.load(Ordering::Relaxed)),
            )
            .finish()
    }
}

#[derive(Serialize, Deserialize)]
pub struct MCTS<P: Player, A: Action, S: State<P, A>> {
    state_map: DashMap<S, StateNode<A>>,
    info: MCTSInfo,
    _marker: PhantomData<(P, A)>,
}

// https://stackoverflow.com/questions/26998485/rust-print-format-number-with-thousand-separator
fn thousands_separate(x: u64) -> String {
    let s = format!("{}", x);
    let bytes: Vec<_> = s.bytes().rev().collect();
    let chunks: Vec<_> = bytes
        .chunks(3)
        .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
        .collect();
    let result: Vec<_> = chunks.join(",").bytes().rev().collect();
    String::from_utf8(result).unwrap()
}
