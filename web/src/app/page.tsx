import Link from "next/link";

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
  model_version: string;
  generated_at: string;
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
    return Array.from(new Set([2026, ...historical])).sort((a, b) => b - a);
  } catch {
    return [2026, 2025, 2024, 2023, 2022, 2021];
  }
}

function TeamMark({ team }: { team: string }) {
  return <span className="team-mark">{team}</span>;
}

function PredictionCard({ game, week }: { game: Prediction; week: number }) {
  return (
    <article className="game-card">
      <div className="game-meta">
        <span>Week {week}</span>
        <span>{game.model_version.replaceAll("-", " ")}</span>
      </div>
      <div className="matchup">
        <div className="team">
          <TeamMark team={game.away_team} />
          <div>
            <span className="team-label">Away</span>
            <strong>{game.away_team}</strong>
          </div>
          <b>{game.predicted_away_score}</b>
        </div>
        <div className="at">@</div>
        <div className="team">
          <TeamMark team={game.home_team} />
          <div>
            <span className="team-label">Home</span>
            <strong>{game.home_team}</strong>
          </div>
          <b>{game.predicted_home_score}</b>
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
        {game.moneyline_signal && (
          <span className="signal-badge">Experimental signal</span>
        )}
      </div>
    </article>
  );
}

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ season?: string; week?: string }>;
}) {
  const query = await searchParams;
  const selectedSeason = Number(query.season) || 2026;
  const selectedWeek = Math.min(18, Math.max(6, Number(query.week) || 6));
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
          <span className="brand-icon">4</span>
          <span>
            Fourth Down
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
              <Link
                aria-label="Previous week"
                aria-disabled={selectedWeek === 6}
                href={`/?season=${selectedSeason}&week=${previousWeek}`}
              >
                ←
              </Link>
              <span>Week {selectedWeek}</span>
              <Link
                aria-label="Next week"
                aria-disabled={selectedWeek === 18}
                href={`/?season=${selectedSeason}&week=${nextWeek}`}
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
            <button type="submit">View season</button>
          </form>

          {board.count > 0 ? (
            <div className="game-grid">
              {board.predictions.map((game) => (
                <PredictionCard
                  key={game.game_id}
                  game={game}
                  week={selectedWeek}
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
