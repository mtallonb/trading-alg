import numpy as np
import pandas as pd
import pytest

from utils.basic import compute_ranking

SESSIONS = [200, 50, 10]


def ranking_rows(margins: list[float], sessions: list[int] = SESSIONS, seed: int = 1, **overrides) -> pd.DataFrame:
    """One synthetic ranking row per margin; overrides maps column -> list of values per row."""
    rng = np.random.default_rng(seed=seed)
    rows = []
    for i, margin in enumerate(margins):
        price = rng.uniform(1, 100)
        row = {
            'NAME': f'A{i}EUR',
            'LAST_TRADE': i,
            'IBS': 1,
            'BLR': 0,
            'CURR_PRICE': price,
            'AVG_B': price * rng.uniform(0.5, 1.5),
            'AVG_S': price * rng.uniform(0.5, 1.5),
            'MARGIN_A': margin,
            'S_TRADES': i,
            'X_TRADES': i,
        }
        row |= {f'AVG_PRICE_{days}': price * rng.uniform(0.5, 1.5) for days in sessions}
        row |= {f'AVG_VOL_{days}': rng.uniform(1e3, 1e5) for days in sessions}
        rows.append(row)
    df = pd.DataFrame(rows)
    for column, values in overrides.items():
        df[column] = values
    return df


def margin_p_by_name(ranking_df: pd.DataFrame) -> dict[str, float]:
    return dict(zip(ranking_df.NAME, ranking_df.MARGIN_P))


def test_margin_p_is_rank_among_positive_margins():
    ranking_df, _ = compute_ranking(df=ranking_rows(margins=[-10, -50, 100, 4000, 50]), sessions=SESSIONS)
    assert margin_p_by_name(ranking_df) == {'A0EUR': 0, 'A1EUR': 0, 'A2EUR': 2 / 3, 'A3EUR': 1, 'A4EUR': 1 / 3}


def test_all_negative_margins_keep_every_asset():
    ranking_df, _ = compute_ranking(df=ranking_rows(margins=[-10, -50, -100, -200, 20]), sessions=SESSIONS)
    assert len(ranking_df) == 5


def test_constant_column_does_not_empty_the_ranking():
    df = ranking_rows(margins=[10, 50, 100, 200, 20], X_TRADES=[3] * 5)
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS)
    assert len(ranking_df) == 5
    assert (ranking_df.X_TRADES == 0).all()


def test_nan_and_no_sells_assets_are_dropped_and_nan_printed(capsys):
    df = ranking_rows(margins=[10, 50, 100, 200, 20], AVG_S=[1.0, 0.0, 1.0, 1.0, 1.0])
    df.loc[2, 'AVG_VOL_50'] = np.nan
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS)
    assert set(ranking_df.NAME) == {'A0EUR', 'A3EUR', 'A4EUR'}
    out = capsys.readouterr().out
    assert 'A2EUR dropped from ranking, NaN in: AVG_VOL_50, VOL' in out
    assert 'A1EUR' not in out  # no sells: dropped silently


def test_dropped_assets_do_not_change_the_scale():
    df = ranking_rows(margins=[10, 50, 100, 200, 20], AVG_S=[1.0, 1.0, 1.0, 1.0, 0.0])
    ranking_all, _ = compute_ranking(df=df, sessions=SESSIONS)
    ranking_filtered, _ = compute_ranking(df=df[df.AVG_S != 0.0].reset_index(drop=True), sessions=SESSIONS)
    pd.testing.assert_frame_equal(ranking_all, ranking_filtered)


@pytest.mark.parametrize('sessions', [[100, 20], [200, 100, 50, 10], [10, 200]])
def test_any_number_of_sessions(sessions):
    df = ranking_rows(margins=[10, 50, 100], sessions=sessions)
    ranking_df, details_df = compute_ranking(df=df, sessions=sessions)
    assert len(ranking_df) == 3
    assert ranking_df.TREND.between(0, 1).all() and ranking_df.VOL.between(0, 1).all()
    assert [col for col in details_df.columns if col.startswith('AVG_PRICE_')] == [f'AVG_PRICE_{d}' for d in sessions]


def test_two_sessions_vol_is_binary():
    ranking_df, _ = compute_ranking(df=ranking_rows(margins=[10] * 8, sessions=[100, 20]), sessions=[100, 20])
    assert set(ranking_df.VOL) <= {0.0, 1.0}


def test_one_session_is_rejected():
    with pytest.raises(ValueError):
        compute_ranking(df=ranking_rows(margins=[10, 20], sessions=[10]), sessions=[10])
