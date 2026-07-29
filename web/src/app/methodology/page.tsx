import Link from "next/link";

const seasons = [
  { year: 2021, games: 523, mae: 11.93, win: 63.0, roi: -17.7 },
  { year: 2022, games: 715, mae: 9.04, win: 62.8, roi: 6.5 },
  { year: 2023, games: 906, mae: 9.95, win: 62.6, roi: -8.4 },
  { year: 2024, games: 1096, mae: 10.09, win: 68.9, roi: 13.7 },
  { year: 2025, games: 1286, mae: 10.51, win: 65.8, roi: -2.0 },
];

export default function Methodology() {
  return (
    <>
      <header className="site-header">
        <Link href="/" className="brand">
          <span className="brand-icon">4</span>
          <span>Fourth Down<small>Forecast</small></span>
        </Link>
        <nav>
          <Link href="/">Predictions</Link>
          <Link className="active" href="/methodology">Methodology</Link>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API ↗</a>
        </nav>
        <span className="model-status"><i /> Model online</span>
      </header>

      <main className="method-page">
        <section className="method-hero">
          <span className="overline">Open methodology</span>
          <h1>A model you can <em>inspect.</em></h1>
          <p>
            The forecast uses two modeling stages, chronological tuning, and
            season-by-season evaluation. Favorable and unfavorable results are
            published together.
          </p>
        </section>

        <section className="metric-row">
          <article><span>Architecture</span><strong>2-stage</strong><small>XGBoost regression</small></article>
          <article><span>Production data</span><strong>8 seasons</strong><small>2018 through 2025</small></article>
          <article><span>Rolling ML bets</span><strong>280</strong><small>Five test seasons</small></article>
          <article><span>Pooled ML ROI</span><strong>+0.74%</strong><small>Experimental, flat stake</small></article>
        </section>

        <section className="model-flow">
          <div className="section-heading">
            <div><span className="overline">How it works</span><h2>From plays to points</h2></div>
          </div>
          <div className="flow-grid">
            <article><b>01</b><h3>Pregame context</h3><p>Five-game rolling efficiency, volume, personnel entropy, opponent strength, and environment.</p></article>
            <article><b>02</b><h3>Game components</h3><p>Stage one estimates passing, rushing, pressure, and expected play volume for both teams.</p></article>
            <article><b>03</b><h3>Score forecast</h3><p>Stage two converts out-of-fold component estimates into expected offensive points.</p></article>
            <article><b>04</b><h3>Simulation</h3><p>Ten thousand seeded score simulations create a winner and model-confidence measure.</p></article>
          </div>
        </section>

        <section className="backtest">
          <div className="section-heading">
            <div><span className="overline">Expanding-window backtest</span><h2>Performance by season</h2></div>
            <span className="experimental-pill">Experimental moneyline signal</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Test season</th><th>Training games</th><th>Margin MAE</th><th>Winner accuracy</th><th>Moneyline ROI</th></tr></thead>
              <tbody>
                {seasons.map((row) => (
                  <tr key={row.year}>
                    <td><strong>{row.year}</strong></td>
                    <td>{row.games.toLocaleString()}</td>
                    <td>{row.mae.toFixed(2)}</td>
                    <td>{row.win.toFixed(1)}%</td>
                    <td className={row.roi >= 0 ? "positive" : "negative"}>
                      {row.roi >= 0 ? "+" : ""}{row.roi.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="disclosure">
            The 62.5% threshold was selected after inspecting historical data.
            The pooled return was +0.74%, which is close to break-even and is
            not evidence of guaranteed future profit. The 2026 season will be
            tracked prospectively without changing the threshold.
          </p>
        </section>
      </main>
      <footer><span>Fourth Down Forecast</span><span>Experimental analysis—not financial advice.</span></footer>
    </>
  );
}
