from __future__ import annotations

from vla_fastvlm.data import FrameRecord, TaskData, build_action_chunks


def _frame(episode: int, index: int, value: float) -> FrameRecord:
    return FrameRecord(
        parquet_path="unused.parquet",
        row_index=index,
        task_name="reach-v3",
        task_text="Reach a goal position",
        episode_index=episode,
        frame_index=index,
        timestamp=index / 80,
        state=(0.0, 0.0, 0.0, 1.0),
        action=(value, value, value, 0.0),
        success=False,
    )


def test_action_chunks_stop_at_episode_boundaries() -> None:
    task = TaskData(
        task_name="reach-v3",
        instruction="Reach a goal position",
        frames=[_frame(0, i, float(i)) for i in range(5)]
        + [_frame(1, i, float(100 + i)) for i in range(3)],
    )

    actions, masks, frame_indices = build_action_chunks(task, 4, frame_stride=2)

    assert actions.shape == (5, 4, 4)
    assert masks.tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    assert actions[2, 0, 0] == 4.0
    assert actions[3, 0, 0] == 100.0
    assert frame_indices.tolist() == [0, 2, 4, 5, 7]


def test_action_chunks_select_every_stride() -> None:
    task = TaskData(
        task_name="reach-v3",
        instruction="Reach a goal position",
        frames=[_frame(0, i, float(i)) for i in range(5)],
    )
    actions, masks, _ = build_action_chunks(task, 2, frame_stride=2)

    assert actions[0, 0, 0] == 0.0
    assert actions[1, 0, 0] == 2.0
    assert actions[2, 0, 0] == 4.0
    assert masks[-1].tolist() == [1.0, 0.0]
