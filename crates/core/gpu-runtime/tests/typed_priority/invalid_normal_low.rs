// compile_fail: a real Normal task cannot await a spawned Low typed task.

use gpu_runtime::executor::{ExecutorError, GpuExecutor};
use gpu_runtime::priority::{
    level::{Low, Normal},
    CompletionCell, PriorityClass,
};

static LOW_RESULT: CompletionCell<u32> = CompletionCell::new();
static NORMAL_RESULT: CompletionCell<()> = CompletionCell::new();

pub unsafe fn build_graph(executor: &GpuExecutor) -> Result<(), ExecutorError> {
    let low = executor.spawn_typed(PriorityClass::<Low>::new(), &LOW_RESULT, |_| async { 7 })?;
    let _normal = executor.spawn_typed(
        PriorityClass::<Normal>::new(),
        &NORMAL_RESULT,
        move |normal| async move {
            let mut low = low;
            let _ = normal.wait_for::<Low, u32>(&mut low).await;
        },
    )?;
    Ok(())
}
