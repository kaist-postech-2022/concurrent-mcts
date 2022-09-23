# concurrent-mcts

This is a library that the user can implement simple concurrent MCTS with small locking.
This archived version is used on [KAPO Science War(KAIST-POSTECH 학생대제전)](https://www.youtube.com/watch?v=N8qPkPN3Bns).
After, it is being developed on [the personal repository](https://github.com/codingskynet/concurrent-mcts).

## Features
- Run MCTS rollouts concurrently(now use locking on creating node, will be improved to lock free).
- Use UCT policy with dynamic coefficient by `choose_weight` on `Action`.
- You can set the rewards on game over(`game_over`) and end of max searching depth(`partial_rewards`)(it can apply on `CyclePolicy::PartialReward`) on `State`.
- You can select cycle policy that allows cycle, ignores cycle and just do it, or use partial rewards.
