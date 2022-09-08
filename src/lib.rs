pub mod mcts;

use std::{
    hash::Hash,
    marker::PhantomData,
    sync::atomic::{AtomicI64, AtomicU64},
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

#[derive(Serialize, Deserialize)]
pub struct MCTS<P: Player, A: Action, S: State<P, A>> {
    #[serde(bound(
        serialize = "DashMap<S, StateNode<A>>: Serialize",
        deserialize = "DashMap<S, StateNode<A>>: Deserialize<'de>"
    ))]
    state_map: DashMap<S, StateNode<A>>,
    _marker: PhantomData<(P, A)>,
}
