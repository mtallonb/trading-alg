import numpy as np
import pandas as pd
import pytest

from utils.basic import compute_ranking

SESSIONS = [200, 50, 10]
TREND_SCALE = 0.1


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
    df = ranking_rows(margins=[-10, -50, 100, 4000, 50])
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)
    assert margin_p_by_name(ranking_df) == {'A0EUR': 0, 'A1EUR': 0, 'A2EUR': 2 / 3, 'A3EUR': 1, 'A4EUR': 1 / 3}


def test_all_negative_margins_keep_every_asset():
    df = ranking_rows(margins=[-10, -50, -100, -200, 20])
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)
    assert len(ranking_df) == 5


def test_constant_column_does_not_empty_the_ranking():
    df = ranking_rows(margins=[10, 50, 100, 200, 20], X_TRADES=[3] * 5)
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)
    assert len(ranking_df) == 5
    assert (ranking_df.X_TRADES == 0).all()


def test_nan_and_no_sells_assets_are_dropped_and_nan_printed(capsys):
    df = ranking_rows(margins=[10, 50, 100, 200, 20], AVG_S=[1.0, 0.0, 1.0, 1.0, 1.0])
    df.loc[2, 'AVG_VOL_50'] = np.nan
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)
    assert set(ranking_df.NAME) == {'A0EUR', 'A3EUR', 'A4EUR'}
    out = capsys.readouterr().out
    assert 'A2EUR dropped from ranking, NaN in: AVG_VOL_50, VOL' in out
    assert 'A1EUR' not in out  # no sells: dropped silently


def test_dropped_assets_do_not_change_the_scale():
    df = ranking_rows(margins=[10, 50, 100, 200, 20], AVG_S=[1.0, 1.0, 1.0, 1.0, 0.0])
    ranking_all, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)
    df_filtered = df[df.AVG_S != 0.0].reset_index(drop=True)
    ranking_filtered, _ = compute_ranking(df=df_filtered, sessions=SESSIONS, trend_scale=TREND_SCALE)
    pd.testing.assert_frame_equal(ranking_all, ranking_filtered)


@pytest.mark.parametrize('sessions', [[100, 20], [200, 100, 50, 10], [10, 200]])
def test_any_number_of_sessions(sessions):
    df = ranking_rows(margins=[10, 50, 100], sessions=sessions)
    ranking_df, details_df = compute_ranking(df=df, sessions=sessions, trend_scale=TREND_SCALE)
    assert len(ranking_df) == 3
    assert ranking_df.TREND.between(0, 1).all() and ranking_df.VOL.between(0, 1).all()
    assert [col for col in details_df.columns if col.startswith('AVG_PRICE_')] == [f'AVG_PRICE_{d}' for d in sessions]


def test_two_sessions_vol_is_binary():
    df = ranking_rows(margins=[10] * 8, sessions=[100, 20])
    ranking_df, _ = compute_ranking(df=df, sessions=[100, 20], trend_scale=TREND_SCALE)
    assert set(ranking_df.VOL) <= {0.0, 1.0}


def test_single_asset_gets_a_ranking_instead_of_nan():
    ranking_df, _ = compute_ranking(df=ranking_rows(margins=[10]), sessions=SESSIONS, trend_scale=TREND_SCALE)
    assert ranking_df.RANKING.tolist() == [0.0]


def test_ranking_is_scaled_to_0_10():
    df = ranking_rows(margins=[10, 50, 100, 200, 20])
    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)
    assert ranking_df.RANKING.max() == 10 and ranking_df.RANKING.min() == 0


@pytest.mark.parametrize('price', [0.1, 0.5, 70000.0])
def test_price_and_volume_equal_to_every_average_are_neutral(price):
    # Float rounding used to give inf (-> 0.5) for 0.1 but 0/0 (asset dropped) for 0.5
    df = ranking_rows(margins=[10, 50, 100])
    for column in ['CURR_PRICE', *[f'AVG_PRICE_{days}' for days in SESSIONS]]:
        df.loc[0, column] = price
    for column in [f'AVG_VOL_{days}' for days in SESSIONS]:
        df.loc[0, column] = price * 1000

    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)

    row = ranking_df[ranking_df.NAME == 'A0EUR']
    assert row.TREND.tolist() == [0.5] and row.VOL.tolist() == [0.5]


@pytest.mark.parametrize(
    ('averages', 'trend'),
    [
        ((61.23, 64.48, 70.07), 1.0),  # BTC: rising from the 200 to the 10-day average
        ((0.28, 0.29, 0.30), 1.0),  # TRX: rising, however small the gaps
        ((1.0, 0.9, 0.8), 0.0),  # falling
        ((0.22, 0.19, 0.22), 0.69),  # SNX: V, 10-day +15.8 % over the 50-day, 10 = 200 (agreement 1/4)
        ((272.19, 208.42, 225.18), 0.60),  # BCH: 10-day +8 % over the 50-day but still below the 200 (agreement 0)
        ((1.0, 1.3, 1.2), 0.41),  # pullback: 10-day -7.7 % under the 50-day, both above the 200
        ((1.0, 1.2, 0.9), 0.26),  # inverted V: 10-day -25 % under the 50-day and below the 200
        ((0.20, 0.22, 0.22), 0.5),  # 10 = 50: no lead
        ((0.22, 0.22, 0.22), 0.5),  # equal
    ],
)
def test_trend_is_led_by_the_10_vs_50_day_averages(averages, trend):
    df = ranking_rows(margins=[10, 50, 100])
    for days, average in zip(SESSIONS, averages):  # SESSIONS = [200, 50, 10]
        df.loc[0, f'AVG_PRICE_{days}'] = average

    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)

    assert ranking_df[ranking_df.NAME == 'A0EUR'].TREND.tolist() == [pytest.approx(trend, abs=0.01)]


def test_only_a_perfect_order_reaches_1_or_0():
    # A huge 10 vs 50-day gap in a mixed case stays inside (0.25, 0.75)
    df = ranking_rows(margins=[10, 50, 100])
    for days, average in zip(SESSIONS, (2.0, 1.0, 3.0)):
        df.loc[0, f'AVG_PRICE_{days}'] = average

    ranking_df, _ = compute_ranking(df=df, sessions=SESSIONS, trend_scale=TREND_SCALE)

    assert 0.25 < ranking_df[ranking_df.NAME == 'A0EUR'].TREND.iloc[0] < 0.75


@pytest.mark.parametrize(
    ('sessions', 'averages', 'trend'),
    [
        ([10, 200], (1.1, 1.0), 1.0),  # order of the sessions list doesn't matter
        ([200, 100, 50, 10], (1.0, 1.1, 1.2, 1.3), 1.0),
        ([10, 200], (1.0, 1.0), 0.5),  # 2 sessions: mixed only when equal
        # One step down (100 > 50) breaks the rise: 10 vs 50 +23.8 %, 4 of the 5 other pairs agree
        ([200, 100, 50, 10], (1.0, 1.1, 1.05, 1.3), 0.74),
    ],
)
def test_trend_with_other_sessions(sessions, averages, trend):
    df = ranking_rows(margins=[10, 50], sessions=sessions)
    for days, average in zip(sessions, averages):
        df.loc[0, f'AVG_PRICE_{days}'] = average

    ranking_df, _ = compute_ranking(df=df, sessions=sessions, trend_scale=TREND_SCALE)

    assert ranking_df[ranking_df.NAME == 'A0EUR'].TREND.tolist() == [pytest.approx(trend, abs=0.01)]


def test_one_session_is_rejected():
    with pytest.raises(ValueError):
        compute_ranking(df=ranking_rows(margins=[10, 20], sessions=[10]), sessions=[10], trend_scale=TREND_SCALE)
