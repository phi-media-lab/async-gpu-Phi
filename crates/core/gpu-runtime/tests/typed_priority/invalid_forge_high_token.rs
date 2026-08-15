// compile_fail: executor-issued current-task tokens have no public constructor.

use gpu_runtime::priority::{level::High, PriorityToken};

pub fn forge_high_token() {
    let _forged = PriorityToken::<High>::from_spawn(7);
}
