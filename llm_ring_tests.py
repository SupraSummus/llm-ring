from llm_ring import get_symmetric_examples, parallel_output_replacement


def test_get_symmetric_examples():
    pairs = [('A', '1'), ('B', '2'), ('C', '3'), ('D', '4')]

    assert get_symmetric_examples(pairs, 0) == [('B', '2'), ('C', '3'), ('D', '4')]
    assert get_symmetric_examples(pairs, 2) == [('D', '4'), ('A', '1'), ('B', '2')]


def test_parallel_output_replacement(monkeypatch):
    pairs = [('A', '1'), ('B', '2'), ('C', '3')]

    monkeypatch.setattr('llm_ring.llm_function', lambda *args: f'Output for {args[1]}')
    result_cw = list(parallel_output_replacement(pairs))
    expected_cw = [
        ('A', 'Output for A'),
        ('B', 'Output for B'),
        ('C', 'Output for C')
    ]
    assert result_cw == expected_cw
