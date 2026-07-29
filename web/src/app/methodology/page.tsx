import Link from "next/link";
import Image from "next/image";

const seasons = [
  { year: 2022, games: 715, predicted: 71, mae: 9.04, win: 62.8, roi: 6.5 },
  { year: 2023, games: 906, predicted: 33, mae: 9.95, win: 62.6, roi: -8.4 },
  { year: 2024, games: 1096, predicted: 65, mae: 10.09, win: 68.9, roi: 13.7 },
  { year: 2025, games: 1286, predicted: 70, mae: 10.51, win: 65.8, roi: -2.0 },
];

const featureGroups = [
  {
    title: "Efficiency",
    detail:
      "Passing EPA, rushing EPA, success rate, yards per pass, and yards per rush.",
  },
  {
    title: "Volume & style",
    detail:
      "Pass and rush volume, pass rate over expectation, time to throw, formation usage, and personnel entropy.",
  },
  {
    title: "Opponent context",
    detail:
      "The same rolling efficiency and volume measures allowed by each defense, plus pressure and blitz tendencies.",
  },
  {
    title: "Game context",
    detail:
      "Separate home and away modeling, wind, recent scoring, points allowed, and strictly lagged market history.",
  },
];

export default function Methodology() {
  return (
    <>
      <header className="site-header">
        <Link href="/" className="brand">
          <Image
            className="brand-mark"
            src="/brand-mark.svg"
            alt="Fourth Down Forecast"
            width={40}
            height={46}
            priority
          />
          <span className="brand-name"><span className="brand-ordinal">th</span> Down<small>Forecast</small></span>
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
          <article><span>Rolling ML bets</span><strong>239</strong><small>2022–2025 showcase</small></article>
          <article><span>Pooled ML ROI</span><strong>+3.90%</strong><small>2022–2025, flat stake</small></article>
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

        <section className="technical-section">
          <div className="section-heading">
            <div>
              <span className="overline">Feature engineering</span>
              <h2>What the model sees</h2>
            </div>
          </div>
          <p className="section-intro">
            Every input is available before kickoff. Team form is calculated
            from the previous five completed games in the same season; the
            current game’s result never enters its own feature row.
          </p>
          <div className="feature-grid">
            {featureGroups.map((group) => (
              <article key={group.title}>
                <h3>{group.title}</h3>
                <p>{group.detail}</p>
              </article>
            ))}
          </div>
          <div className="technical-note">
            <strong>Why predictions begin in Week 6</strong>
            <p>
              Version 1 requires five completed current-season games for both
              teams. It does not silently carry prior-season form across roster
              and coaching changes. An offseason-carryover approach would be a
              separate model requiring its own backtest.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <div className="section-heading">
            <div>
              <span className="overline">Model architecture</span>
              <h2>Two stages, one score forecast</h2>
            </div>
          </div>
          <div className="detail-columns">
            <article>
              <span className="detail-number">Stage 01</span>
              <h3>Predict game components</h3>
              <p>
                Fourteen XGBoost regressors estimate home and away passing EPA,
                rushing EPA, yards per pass, yards per rush, pass plays, rush
                plays, and defensive pressure rate.
              </p>
              <p>
                Stage-one training predictions are generated out of fold using
                five-fold shuffled K-fold stacking. A row is never predicted by
                a model fitted on that same row, although a fold can use games
                that occurred later in the training period.
              </p>
            </article>
            <article>
              <span className="detail-number">Stage 02</span>
              <h3>Translate components into points</h3>
              <p>
                Two additional XGBoost regressors combine the out-of-fold game
                components with recent scoring, opponent scoring allowed, EPA,
                touchdown and field-goal differentials, and lagged market form.
              </p>
              <p>
                The outputs are expected home and away offensive points. The
                displayed predicted score is produced from the subsequent
                simulation rather than copied from a sportsbook line.
              </p>
            </article>
          </div>
        </section>

        <section className="technical-section training-panel">
          <div>
            <span className="overline">Training discipline</span>
            <h2>How parameters are selected</h2>
          </div>
          <div className="training-list">
            <article>
              <strong>Chronological tuning</strong>
              <p>
                Hyperparameters are selected with TimeSeriesSplit so tuning
                folds train on earlier games and validate on later games.
              </p>
            </article>
            <article>
              <strong>Expanding evaluation</strong>
              <p>
                Each historical season uses a newly fitted model containing
                only seasons that had already finished. For example, the 2025
                page uses a model trained through 2024.
              </p>
            </article>
            <article>
              <strong>Production refit</strong>
              <p>
                After model choices were frozen, the production artifact was
                refitted on every completed season from 2018–2025. The next
                genuinely prospective evaluation period is 2026.
              </p>
            </article>
          </div>
        </section>

        <section className="technical-section split-section">
          <div className="section-heading">
            <div>
              <span className="overline">Validation design</span>
              <h2>Why K-fold and time-series splits are both used</h2>
            </div>
          </div>
          <p className="section-intro">
            The two split methods solve different problems. Their roles were
            tested separately across independent 2022–2025 seasons rather than
            chosen from a single favorable year.
          </p>
          <div className="split-grid">
            <article>
              <span className="detail-number">Stage-one stacking</span>
              <h3>Shuffled K-fold</h3>
              <p>
                Stage two must learn from stage-one estimates created without
                fitting stage one on the same game. Shuffled K-fold supplies an
                out-of-fold estimate for every training row and preserves more
                data for stage-two fitting.
              </p>
              <p>
                This is not perfectly chronological. A fold predicting an
                earlier game may contain later games, creating temporal
                information leakage inside the training process. The leaked
                information never includes the held-out test season, but it can
                make the stage-two training inputs more representative than
                they would have been in real time.
              </p>
            </article>
            <article>
              <span className="detail-number">Hyperparameter tuning</span>
              <h3>TimeSeriesSplit</h3>
              <p>
                Parameter searches train on earlier rows and validate on later
                rows. This more closely matches deployment and prevents the
                tuning procedure from selecting settings using future games
                within its validation folds.
              </p>
              <p>
                Replacing this step with shuffled K-fold reduced 2022–2025
                moneyline ROI from 3.90% to 0.16% and slightly reduced winner
                accuracy, despite improving margin MAE by only 0.05 points.
              </p>
            </article>
          </div>
          <div className="technical-note">
            <strong>Why K-fold stacking was retained</strong>
            <p>
              We also replaced stage-one K-fold stacking with chronological
              splits while leaving the outer seasonal tests untouched. The
              chronological version produced 64.13% winner accuracy and 2.00%
              moneyline ROI; K-fold produced 65.05% and 3.90%. Because every
              reported season remained completely outside its model’s training
              data, this is evidence of better independent-season performance,
              not proof that the internal leakage is harmless. The limitation
              is disclosed and the 2026 predictions remain the prospective
              test.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <div className="section-heading">
            <div>
              <span className="overline">Simulation & confidence</span>
              <h2>From expected points to a winner</h2>
            </div>
          </div>
          <div className="simulation-copy">
            <p>
              Expected offensive points are converted into touchdown and
              field-goal scoring rates. Ten thousand seeded Poisson simulations
              are run for each matchup, including a small non-offensive
              touchdown component. Median simulated scores become the displayed
              forecast.
            </p>
            <p>
              “Model confidence” is the share of simulations won by the
              predicted team. It is useful for ranking conviction, but it has
              not been proven to be a perfectly calibrated probability. That
              is why the site does not describe a 65% confidence value as a
              literal 65% chance of winning.
            </p>
          </div>
        </section>

        <section className="backtest">
          <div className="section-heading">
            <div><span className="overline">Expanding-window backtest</span><h2>Performance by season</h2></div>
            <span className="experimental-pill">Experimental moneyline signal</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Test season</th><th>Training games</th><th>Games predicted<br /><small>≥62.5% confidence</small></th><th>Margin MAE</th><th>Winner accuracy</th><th>Moneyline ROI</th></tr></thead>
              <tbody>
                {seasons.map((row) => (
                  <tr key={row.year}>
                    <td><strong>{row.year}</strong></td>
                    <td>{row.games.toLocaleString()}</td>
                    <td>{row.predicted}</td>
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
            The public showcase begins in 2022, when the expanding training
            window contained 715 games. The earlier 2021 test remains preserved
            in the full research report but is excluded from this displayed
            aggregate because its training history was materially smaller. The
            62.5% threshold was selected retrospectively, so the +3.90% return
            is not evidence of guaranteed future profit. The 2026 season will
            be tracked prospectively without changing the threshold.
          </p>
        </section>

        <section className="technical-section decision-section">
          <div className="section-heading">
            <div>
              <span className="overline">Product decisions</span>
              <h2>What did not make the cut</h2>
            </div>
          </div>
          <div className="decision-grid">
            <article>
              <span>Removed</span>
              <h3>Spread indicator</h3>
              <p>
                The four-point spread rule looked promising in 2024–2025 but
                returned −5.43% across the full five-season rolling backtest.
                It is preserved in research and excluded from the product.
              </p>
            </article>
            <article>
              <span>Not promoted</span>
              <h3>Totals indicator</h3>
              <p>
                Every predeclared totals threshold lost during 2022–2023
                development. A six-point pocket worked later, but failed the
                stability requirement and was not added to the interface.
              </p>
            </article>
            <article>
              <span>Experimental</span>
              <h3>Moneyline signal</h3>
              <p>
                A predicted winner is flagged at 62.5% model confidence. The
                public 2022–2025 showcase returned +3.90% across 239 bets, but
                the threshold was chosen retrospectively and must prove itself
                prospectively in 2026.
              </p>
            </article>
          </div>
        </section>

        <section className="limitations">
          <span className="overline">Read before interpreting</span>
          <h2>Known limitations</h2>
          <ul>
            <li>Injuries and confirmed quarterback availability are not explicit model inputs.</li>
            <li>Historical betting comparisons use closing lines; a live prediction may capture a different price.</li>
            <li>NFL seasons are small samples, and a few underdog outcomes can materially change moneyline ROI.</li>
            <li>The public 2022–2025 aggregate excludes the smaller 2021 training window; that result remains in the repository.</li>
            <li>Historical performance, including profitable periods, does not guarantee future returns.</li>
          </ul>
        </section>
      </main>
      <footer><span>Fourth Down Forecast</span><span>Experimental analysis—not financial advice.</span></footer>
    </>
  );
}
