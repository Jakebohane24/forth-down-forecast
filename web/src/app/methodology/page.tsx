import Link from "next/link";
import Image from "next/image";

const API_URL = process.env.API_URL ?? "http://localhost:8000";

type PerformanceSeason = {
  season: number;
  training_games: number;
  prediction_count: number;
  evaluated_games: number;
  margin_mae: number | null;
  win_accuracy: number | null;
  moneyline_bets: number;
  moneyline_settled: number;
  moneyline_accuracy: number | null;
  moneyline_roi: number | null;
  standard_tier_bets?: number;
  standard_tier_accuracy?: number | null;
  standard_tier_roi?: number | null;
  high_tier_bets?: number;
  high_tier_accuracy?: number | null;
  high_tier_roi?: number | null;
};

const fallbackSeasons: PerformanceSeason[] = [
  { season: 2022, training_games: 715, prediction_count: 191, evaluated_games: 191, moneyline_bets: 48, moneyline_settled: 48, moneyline_accuracy: .646, margin_mae: 9.07, win_accuracy: .613, moneyline_roi: -.024, standard_tier_bets: 34, standard_tier_accuracy: .588, standard_tier_roi: -.100, high_tier_bets: 14, high_tier_accuracy: .786, high_tier_roi: .160 },
  { season: 2023, training_games: 906, prediction_count: 190, evaluated_games: 190, moneyline_bets: 37, moneyline_settled: 37, moneyline_accuracy: .676, margin_mae: 9.92, win_accuracy: .637, moneyline_roi: .058, standard_tier_bets: 24, standard_tier_accuracy: .625, standard_tier_roi: .019, high_tier_bets: 13, high_tier_accuracy: .769, high_tier_roi: .131 },
  { season: 2024, training_games: 1096, prediction_count: 190, evaluated_games: 190, moneyline_bets: 52, moneyline_settled: 52, moneyline_accuracy: .712, margin_mae: 9.98, win_accuracy: .732, moneyline_roi: .074, standard_tier_bets: 35, standard_tier_accuracy: .657, standard_tier_roi: -.009, high_tier_bets: 17, high_tier_accuracy: .824, high_tier_roi: .245 },
  { season: 2025, training_games: 1286, prediction_count: 190, evaluated_games: 190, moneyline_bets: 37, moneyline_settled: 37, moneyline_accuracy: .649, margin_mae: 10.45, win_accuracy: .626, moneyline_roi: .056, standard_tier_bets: 28, standard_tier_accuracy: .679, standard_tier_roi: .116, high_tier_bets: 9, high_tier_accuracy: .556, high_tier_roi: -.130 },
  { season: 2026, training_games: 1476, prediction_count: 0, evaluated_games: 0, moneyline_bets: 0, moneyline_settled: 0, moneyline_accuracy: null, margin_mae: null, win_accuracy: null, moneyline_roi: null },
];

async function getPerformance(): Promise<PerformanceSeason[]> {
  try {
    const response = await fetch(`${API_URL}/performance`, {
      next: { revalidate: 300 },
    });
    if (!response.ok) throw new Error("Performance API unavailable");
    const data = await response.json();
    return data.seasons;
  } catch {
    return fallbackSeasons;
  }
}

const featureGroups = [
  {
    title: "Play efficiency",
    features: [
      "Overall EPA per play: the average change in expected points created by the offense.",
      "Passing EPA and rushing EPA: efficiency split by play type.",
      "Success rate: the share of plays that produce positive EPA.",
      "Completion percentage (CP): completed passes divided by pass attempts with a recorded completion result across the five-game window.",
      "Completion percentage over expected (CPOE): nflverse's play-level completion performance relative to expected difficulty, averaged across recent attempts.",
    ],
  },
  {
    title: "Pace & passing tendency",
    features: [
      "Average pass plays: recent passing volume.",
      "Pass rate: passes as a share of scrimmage plays.",
      "Pass rate over expectation (PROE): how often a team passes relative to situation-based expectations.",
      "Time to throw: average time between the snap and the quarterback releasing the ball.",
      "Opponent pass-play volume: the average number of passes recently faced by the opposing defense.",
    ],
  },
  {
    title: "Formation & personnel",
    features: [
      "Shotgun/spread rate: the share of charted plays run from shotgun or empty formations.",
      "Heavy-formation rate: the share run from singleback, under-center, I-formation, or jumbo looks.",
      "Average defenders in the box: a measure of defensive run-front density.",
      "Offensive personnel entropy: Shannon entropy of recent offensive lineup combinations.",
      "Defensive personnel entropy: Shannon entropy of recent defensive lineup combinations. Higher entropy means greater lineup variety.",
    ],
  },
  {
    title: "Defense & opponent strength",
    features: [
      "Pressure rate: how often the quarterback was pressured.",
      "Blitz rate: how frequently the defense sent extra pass rushers.",
      "Opponent-adjusted passing EPA and rushing EPA: the recent efficiency allowed by the opposing unit.",
      "Opponent-adjusted success rate: the recent successful-play rate allowed by the opponent.",
      "Opponent-adjusted pressure rate: the pressure tendency associated with the opposing matchup.",
    ],
  },
  {
    title: "Scoring form",
    features: [
      "Average offensive points scored and average offensive points allowed.",
      "Offensive point differential: points scored minus points allowed over the five-game window.",
      "Offensive touchdown differential: touchdowns scored minus offensive touchdowns allowed.",
      "Field-goal differential: field goals made minus field goals allowed.",
      "Every measure is maintained separately for the home and away team.",
    ],
  },
  {
    title: "Lagged market context",
    features: [
      "Average points above spread: recent scoring margin relative to the historical closing spread.",
      "Average result-plus-spread: the retained companion encoding of recent results and closing lines.",
      "Home-away difference in points above spread.",
      "Home-away difference in result-plus-spread.",
      "Only prior games enter these fields; the current game's spread and moneyline are not score-model inputs.",
    ],
  },
];

export default async function Methodology() {
  const seasons = await getPerformance();
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
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">API ↗</a>
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
          <article><span>Rolling ML signals</span><strong>174</strong><small>2022–2025 showcase</small></article>
          <article><span>Pooled signal ROI</span><strong>+3.99%</strong><small>2022–2025, flat stake</small></article>
        </section>

        <section className="model-flow">
          <div className="section-heading">
            <div><span className="overline">How it works</span><h2>From plays to points</h2></div>
          </div>
          <div className="flow-grid">
            <article><b>01</b><h3>Pregame context</h3><p>Five-game rolling efficiency, volume, personnel entropy, opponent strength, and market context.</p></article>
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
            current game’s result never enters its own feature row. Unless a
            difference is named explicitly, each measure is calculated for
            both the home and away team.
          </p>
          <div className="feature-grid">
            {featureGroups.map((group) => (
              <article key={group.title}>
                <h3>{group.title}</h3>
                <ul>
                  {group.features.map((feature) => (
                    <li key={feature}>{feature}</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
          <div className="technical-note">
            <strong>Stage-one generated features</strong>
            <p>
              The inputs above feed seven component forecasts for each team:
              passing EPA, rushing EPA, passing yards per play, rushing yards
              per attempt, pass-play volume, rush-play volume, and pressure
              rate. Stage two uses those generated estimates alongside scoring
              form and lagged market context. They are intermediate model
              outputs, not additional raw pregame statistics.
            </p>
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
          <div className="technical-note">
            <strong>Weather display policy</strong>
            <p>
              Open-Meteo supplies an hourly forecast for the stadium at
              kickoff. It is displayed as supplemental game context but is not
              an input to the official model. This keeps historical and live
              inference consistent while conditions can refresh without
              rewriting the locked prediction.
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
                In the controlled confidence-only research comparison,
                replacing this step with shuffled K-fold reduced 2022–2025
                signal ROI from 3.90% to 0.16% and slightly reduced winner
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
              ROI under the confidence-only research rule; K-fold produced
              65.05% and 3.90%. Because every reported season remained
              completely outside its model’s training data, this is evidence
              of better independent-season performance, not proof that the
              internal leakage is harmless. The limitation is disclosed and
              the 2026 predictions remain the prospective test.
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
              forecast. When a simulation ends level, half a win is allocated
              to each team to approximate an unresolved overtime outcome rather
              than incorrectly crediting either side with every simulated tie.
            </p>
            <p>
              “Model confidence” is the share of simulations won by the
              predicted team after that split-tie adjustment. The model pick is
              based on the full joint simulation distribution, while the two
              displayed scores are separate medians; in a small number of games,
              those summaries can point in opposite directions. Confidence is
              useful for ranking conviction, but it has
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
              <thead><tr><th>Test season</th><th>Training games</th><th>Official signals<br /><small>Confidence 60%+; ML −300 or better</small></th><th>Signal accuracy</th><th>Margin MAE</th><th>Winner accuracy</th><th>Signal ROI</th></tr></thead>
              <tbody>
                {seasons.map((row) => (
                  <tr key={row.season} className={row.season === 2026 ? "prospective-season" : undefined}>
                    <td>
                      <strong>{row.season}</strong>
                      {row.season === 2026 && <small>Prospective</small>}
                    </td>
                    <td>{row.training_games.toLocaleString()}</td>
                    <td>{row.prediction_count > 0 ? row.moneyline_bets : "Pending"}</td>
                    <td>{row.moneyline_accuracy === null ? "—" : `${(row.moneyline_accuracy * 100).toFixed(1)}%`}</td>
                    <td>{row.margin_mae === null ? "—" : row.margin_mae.toFixed(2)}</td>
                    <td>{row.win_accuracy === null ? "—" : `${(row.win_accuracy * 100).toFixed(1)}%`}</td>
                    <td className={row.moneyline_roi === null ? undefined : row.moneyline_roi >= 0 ? "positive" : "negative"}>
                      {row.moneyline_roi === null
                        ? "—"
                        : `${row.moneyline_roi >= 0 ? "+" : ""}${(row.moneyline_roi * 100).toFixed(1)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="section-heading confidence-tier-heading">
            <div>
              <span className="overline">Prospective tier tracking</span>
              <h3>How confidence bands behave</h3>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Season</th>
                  <th>60%–under 65%<br /><small>Signals · accuracy · ROI</small></th>
                  <th>65%+ high confidence<br /><small>Signals · accuracy · ROI</small></th>
                </tr>
              </thead>
              <tbody>
                {seasons.map((row) => (
                  <tr key={`tier-${row.season}`} className={row.season === 2026 ? "prospective-season" : undefined}>
                    <td><strong>{row.season}</strong>{row.season === 2026 && <small>Prospective</small>}</td>
                    <td>
                      {(row.standard_tier_bets ?? 0) > 0
                        ? `${row.standard_tier_bets} · ${((row.standard_tier_accuracy ?? 0) * 100).toFixed(1)}% · ${((row.standard_tier_roi ?? 0) * 100) >= 0 ? "+" : ""}${((row.standard_tier_roi ?? 0) * 100).toFixed(1)}%`
                        : "Pending"}
                    </td>
                    <td>
                      {(row.high_tier_bets ?? 0) > 0
                        ? `${row.high_tier_bets} · ${((row.high_tier_accuracy ?? 0) * 100).toFixed(1)}% · ${((row.high_tier_roi ?? 0) * 100) >= 0 ? "+" : ""}${((row.high_tier_roi ?? 0) * 100).toFixed(1)}%`
                        : "Pending"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="technical-note">
            <strong>Uncertainty around the +3.99% historical ROI</strong>
            <p>
              The 95% Student&apos;s t interval is −7.30% to +15.28%. An
              ordinary bootstrap with 100,000 resamples gives a
              normal-approximation interval of −7.19% to +15.16%. Both include
              zero, so the historical sample does not establish that the
              signal&apos;s underlying mean return is positive. The intervals
              exclude zero only below approximately 51.3% confidence for the
              t method and 51.6% for the bootstrap method.
            </p>
          </div>
          <div className="technical-note">
            <strong>Uncertainty around signal accuracy</strong>
            <p>
              The signal won 117 of 174 games, an observed accuracy of 67.2%.
              Its 95% Wilson interval is 60.0% to 73.8%, expressing the
              sampling uncertainty around the underlying win proportion.
            </p>
          </div>
          <p className="disclosure">
            The public showcase begins in 2022, when the expanding training
            window contained 715 games. The earlier 2021 test remains preserved
            in the full research report but is excluded from this displayed
            aggregate because its training history was materially smaller. The
            combined 60% confidence and −300 price floor was selected
            retrospectively after a tie-handling correction and threshold
            sensitivity analysis. It favors a larger sample and steadier
            season-level returns over the higher but more volatile historical
            ROI of the 65% subgroup. Therefore, the +3.99% return is not
            evidence of guaranteed future profit. The 2026 rule will remain
            frozen; 60%–under-65% and 65%+ results will be tracked separately
            and reconsidered only after the complete season.
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
              <span>Removed</span>
              <h3>Wind feature</h3>
              <p>
                Removing wind slightly improved margin MAE while preserving
                65.05% winner accuracy across the 2022–2025 rolling tests.
                Stadium weather remains visible as context, but it is excluded
                from the model because historical final wind and a live
                pregame forecast are not equivalent inputs.
              </p>
            </article>
          </div>
        </section>

        <section className="technical-section production-signal">
          <div className="section-heading">
            <div>
              <span className="overline">Included in the product</span>
              <h2>Experimental production signal</h2>
            </div>
          </div>
          <article>
            <span>Officially tracked</span>
            <h3>Moneyline signal</h3>
            <p>
              A predicted winner is flagged only at 60% model confidence
              or higher and a selected-team moneyline of −300 or better. The
              signal is displayed in the live product and its picks lock one
              hour before kickoff. The public 2022–2025 showcase returned
              +3.99% across 174 signals. Signals at 65% or higher receive a
              high-confidence label, allowing the two confidence bands to be
              monitored without changing the official wager rule. Both
              conditions were selected retrospectively and must prove
              themselves prospectively in 2026.
            </p>
          </article>
        </section>

        <section className="limitations">
          <span className="overline">Read before interpreting</span>
          <h2>Known limitations</h2>
          <ul>
            <li>Injuries and confirmed quarterback availability are not explicit model inputs.</li>
            <li>Historical betting comparisons use closing lines; a live prediction may capture a different price.</li>
            <li>Kickoff weather is a forecast rather than a guaranteed field-level measurement, and retractable-roof decisions may not be public when a prediction locks.</li>
            <li>NFL seasons are small samples, and a few underdog outcomes can materially change signal ROI.</li>
            <li>The public 2022–2025 aggregate excludes the smaller 2021 training window; that result remains in the repository.</li>
            <li>Historical performance, including profitable periods, does not guarantee future returns.</li>
          </ul>
        </section>
      </main>
      <footer><span>Fourth Down Forecast</span><span>Experimental analysis—not financial advice.</span></footer>
    </>
  );
}
