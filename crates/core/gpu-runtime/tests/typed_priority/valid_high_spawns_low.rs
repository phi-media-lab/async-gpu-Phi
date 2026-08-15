// should_compile: wait ordering does not restrict fire-and-forget spawn targets.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{High, Low},
    CompletionCell, PriorityClass, PriorityToken,
};

static LOW_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn high_fire_and_forgets_low(
    executor: &GpuExecutor,
    _high_context: &PriorityToken<High>,
) -> Result<(), ExecutorError> {
    let _low = executor.spawn_typed(PriorityClass::<Low>::new(), &LOW_RESULT, |_| async {})?;
    Ok(())
}
