import Link from "next/link";
import Image from "next/image";

type Prediction = {
  game_id: string;
  home_team: string;
  away_team: string;
  predicted_home_score: number;
  predicted_away_score: number;
  predicted_winner: string;
  model_win_confidence: number;
  moneyline_signal: boolean;
  home_moneyline: number | null;
  away_moneyline: number | null;
  moneyline_signal_odds: number | null;
  model_version: string;
  generated_at: string;
  actual_home_score: number | null;
  actual_away_score: number | null;
  prediction_correct: boolean | null;
  moneyline_signal_won: boolean | null;
  moneyline_signal_profit: number | null;
};

type WeekResponse = {
  season: number;
  week: number;
  count: number;
  predictions: Prediction[];
};

const API_URL = process.env.API_URL ?? "http://localhost:8000";

async function getPredictions(season: number, week: number): Promise<WeekResponse> {
  try {
    const response = await fetch(`${API_URL}/predictions/${season}/${week}`, {
      next: { revalidate: 300 },
    });
    if (!response.ok) throw new Error("Prediction API unavailable");
    return response.json();
  } catch {
    return { season, week, count: 0, predictions: [] };
  }
}

async function getSeasons(): Promise<number[]> {
  try {
    const response = await fetch(`${API_URL}/seasons`, {
      next: { revalidate: 300 },
    });
    if (!response.ok) throw new Error("Season API unavailable");
    const historical: number[] = await response.json();
    return Array.from(
      new Set([2026, ...historical.filter((season) => season >= 2022)]),
    ).sort((a, b) => b - a);
  } catch {
    return [2026, 2025, 2024, 2023, 2022];
  }
}

function TeamMark({ team }: { team: string }) {
  return (
    <span className="team-mark">
      <Image
        src={`/team-logos/${team}.png`}
        alt={`${team} team logo`}
        width={34}
        height={34}
      />
    </span>
  );
}

function formatOdds(odds: number | null) {
  if (odds === null) return "—";
  return odds > 0 ? `+${odds}` : `${odds}`;
}

function PredictionCard({
  game,
  week,
  pregamePreview = false,
}: {
  game: Prediction;
  week: number;
  pregamePreview?: boolean;
}) {
  const completed =
    !pregamePreview &&
    game.actual_home_score !== null &&
    game.actual_away_score !== null;
  return (
    <article
      className={`game-card ${
        completed
          ? game.prediction_correct
            ? "pick-correct"
            : "pick-incorrect"
          : ""
      }`}
    >
      <div className="game-meta">
        <span>{completed ? "Final" : `Week ${week}`}</span>
        <span>{game.model_version.replaceAll("-", " ")}</span>
      </div>
      <div className="matchup">
        <div className="team">
          <TeamMark team={game.away_team} />
          <div>
            <span className="team-label">Away</span>
            <strong>{game.away_team}</strong>
            <small className="moneyline">
              ML {formatOdds(game.away_moneyline)}
            </small>
          </div>
          {completed ? (
            <div className="score-comparison">
              <div className="score-column predicted-score">
                <span>Prediction</span>
                <b>{game.predicted_away_score}</b>
              </div>
              <div className="score-column final-score">
                <span>Final</span>
                <b>{game.actual_away_score}</b>
              </div>
            </div>
          ) : (
            <div className="score-block">
              <b>{game.predicted_away_score}</b>
            </div>
          )}
        </div>
        <div className="at">@</div>
        <div className="team">
          <TeamMark team={game.home_team} />
          <div>
            <span className="team-label">Home</span>
            <strong>{game.home_team}</strong>
            <small className="moneyline">
              ML {formatOdds(game.home_moneyline)}
            </small>
          </div>
          {completed ? (
            <div className="score-comparison">
              <div className="score-column predicted-score">
                <span>Prediction</span>
                <b>{game.predicted_home_score}</b>
              </div>
              <div className="score-column final-score">
                <span>Final</span>
                <b>{game.actual_home_score}</b>
              </div>
            </div>
          ) : (
            <div className="score-block">
              <b>{game.predicted_home_score}</b>
            </div>
          )}
        </div>
      </div>
      <div className="prediction-footer">
        <div>
          <span className="eyebrow">Model pick</span>
          <strong>{game.predicted_winner}</strong>
        </div>
        <div>
          <span className="eyebrow">Confidence</span>
          <strong>{(game.model_win_confidence * 100).toFixed(1)}%</strong>
        </div>
        {completed && (
          <span
            className={`result-badge ${
              game.prediction_correct ? "won" : "lost"
            }`}
          >
            Pick {game.prediction_correct ? "correct" : "missed"}
          </span>
        )}
        {game.moneyline_signal && !completed && (
          <span className="signal-badge">
            Experimental signal · {formatOdds(game.moneyline_signal_odds)}
          </span>
        )}
        {game.moneyline_signal && completed && (
          <span
            className={`signal-badge ${
              game.moneyline_signal_won ? "won" : "lost"
            }`}
          >
            Signal {game.moneyline_signal_won ? "won" : "lost"} ·{" "}
            {game.moneyline_signal_profit !== null
              ? `${game.moneyline_signal_profit > 0 ? "+" : ""}${game.moneyline_signal_profit.toFixed(2)}u`
              : "odds unavailable"}
          </span>
        )}
      </div>
    </article>
  );
}

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ season?: string; week?: string; view?: string }>;
}) {
  const query = await searchParams;
  const selectedSeason = Number(query.season) || 2026;
  const selectedWeek = Math.min(18, Math.max(6, Number(query.week) || 6));
  const pregamePreview = query.view === "pregame" && selectedSeason < 2026;
  const viewQuery = pregamePreview ? "&view=pregame" : "";
  const [board, seasons] = await Promise.all([
    getPredictions(selectedSeason, selectedWeek),
    getSeasons(),
  ]);
  const previousWeek = Math.max(6, selectedWeek - 1);
  const nextWeek = Math.min(18, selectedWeek + 1);

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
          <span className="brand-name">
            <span className="brand-ordinal">th</span> Down
            <small>Forecast</small>
          </span>
        </Link>
        <nav>
          <Link className="active" href="/">Predictions</Link>
          <Link href="/methodology">Methodology</Link>
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">API ↗</a>
        </nav>
        <span className="model-status"><i /> Model online</span>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <span className="overline">NFL prediction intelligence</span>
            <h1>See the game before <em>kickoff.</em></h1>
            <p>
              Data-driven score forecasts built from play-level efficiency,
              matchup context, and a rigorously backtested two-stage model.
            </p>
          </div>
          <div className="hero-stats">
            <div>
              <span>Production training</span>
              <strong>2018–25</strong>
            </div>
            <div>
              <span>Historical games</span>
              <strong>1,476</strong>
            </div>
            <div>
              <span>Moneyline trigger</span>
              <strong>62.5%</strong>
            </div>
          </div>
        </section>

        <section className="dashboard">
          <div className="section-heading">
            <div>
              <span className="overline">Weekly board</span>
              <h2>{selectedSeason} predictions</h2>
            </div>
            <div className="week-controls">
              {selectedSeason < 2026 && (
                <Link
                  className="view-toggle"
                  href={
                    pregamePreview
                      ? `/?season=${selectedSeason}&week=${selectedWeek}`
                      : `/?season=${selectedSeason}&week=${selectedWeek}&view=pregame`
                  }
                  scroll={false}
                >
                  {pregamePreview ? "Show final results" : "Preview pregame"}
                </Link>
              )}
              <Link
                aria-label="Previous week"
                aria-disabled={selectedWeek === 6}
                href={`/?season=${selectedSeason}&week=${previousWeek}${viewQuery}`}
                scroll={false}
              >
                ←
              </Link>
              <span>Week {selectedWeek}</span>
              <Link
                aria-label="Next week"
                aria-disabled={selectedWeek === 18}
                href={`/?season=${selectedSeason}&week=${nextWeek}${viewQuery}`}
                scroll={false}
              >
                →
              </Link>
            </div>
          </div>
          <form className="season-picker" action="/">
            <label htmlFor="season">Explore a season</label>
            <select id="season" name="season" defaultValue={selectedSeason}>
              {seasons.map((season) => (
                <option value={season} key={season}>
                  {season}{season === 2026 ? " · prospective" : " · historical"}
                </option>
              ))}
            </select>
            <input type="hidden" name="week" value={selectedWeek} />
            {pregamePreview && <input type="hidden" name="view" value="pregame" />}
            <button type="submit">View season</button>
          </form>

          {board.count > 0 ? (
            <div className="game-grid">
              {board.predictions.map((game) => (
                <PredictionCard
                  key={game.game_id}
                  game={game}
                  week={selectedWeek}
                  pregamePreview={pregamePreview}
                />
              ))}
            </div>
          ) : (
            <div className="empty-board">
              <div className="field-lines" />
              <span className="empty-icon">01</span>
              <h3>Week {selectedWeek} board opens soon</h3>
              <p>
                Predictions will publish after the official schedule, current
                team features, and market prices are available.
              </p>
              <span className="locked-note">No fabricated preseason picks</span>
            </div>
          )}
        </section>

        <section className="trust-strip">
          <div>
            <span className="overline">Built for accountability</span>
            <h2>Every prediction. Frozen before kickoff.</h2>
          </div>
          <p>
            Each weekly forecast is stored with its model version, generation
            time, and available market price—creating an auditable prospective
            record instead of rewriting history.
          </p>
          <Link href="/methodology">Explore the model <span>→</span></Link>
        </section>
      </main>

      <footer>
        <span>Fourth Down Forecast</span>
        <span>Experimental analysis—not financial advice.</span>
      </footer>
    </>
  );
}
