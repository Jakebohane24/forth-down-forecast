import Link from "next/link";
import Image from "next/image";

type Prediction = {
  game_id: string;
  home_team: string;
  away_team: string;
  predicted_home_score: number | null;
  predicted_away_score: number | null;
  predicted_winner: string | null;
  model_win_confidence: number | null;
  moneyline_signal: boolean;
  home_moneyline: number | null;
  away_moneyline: number | null;
  moneyline_signal_odds: number | null;
  model_version: string | null;
  generated_at: string;
  kickoff: string | null;
  actual_home_score: number | null;
  actual_away_score: number | null;
  prediction_correct: boolean | null;
  moneyline_signal_won: boolean | null;
  moneyline_signal_profit: number | null;
  venue_name: string | null;
  venue_type: string | null;
  roof_status: string | null;
  country_code: string | null;
  forecast_for: string | null;
  weather_retrieved_at: string | null;
  wind_mph: number | null;
  wind_gust_mph: number | null;
  temperature_f: number | null;
  precipitation_probability: number | null;
  precipitation_inches: number | null;
  weather_code: number | null;
  weather_source: string | null;
  schedule_only?: boolean;
};

type WeekResponse = {
  season: number;
  week: number;
  count: number;
  predictions: Prediction[];
};

type ScheduleGame = {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  kickoff: string;
  venue_name: string;
  venue_type: string;
  roof_status: string;
  country_code: string;
  prediction_eligible: boolean;
};

type ScheduleResponse = {
  season: number;
  week: number;
  count: number;
  games: ScheduleGame[];
};

const API_URL = process.env.API_URL ?? "http://localhost:8000";

const WEATHER_PREVIEW_2025_WEEK_6: Record<string, Partial<Prediction>> = {
  "2025_06_ARI_IND": {
    venue_name: "Lucas Oil Stadium",
    venue_type: "retractable",
    roof_status: "pending",
    wind_mph: 14,
    wind_gust_mph: 22,
    temperature_f: 72,
    precipitation_probability: 10,
  },
  "2025_06_DAL_CAR": {
    venue_name: "Bank of America Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 5,
    wind_gust_mph: 9,
    temperature_f: 76,
    precipitation_probability: 5,
  },
  "2025_06_DEN_NYJ": {
    venue_name: "MetLife Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 17,
    wind_gust_mph: 28,
    temperature_f: 58,
    precipitation_probability: 75,
  },
  "2025_06_DET_KC": {
    venue_name: "GEHA Field at Arrowhead Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 11,
    wind_gust_mph: 18,
    temperature_f: 64,
    precipitation_probability: 20,
  },
  "2025_06_LAC_MIA": {
    venue_name: "Hard Rock Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 8,
    wind_gust_mph: 13,
    temperature_f: 84,
    precipitation_probability: 45,
  },
  "2025_06_LA_BAL": {
    venue_name: "M&T Bank Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 14,
    wind_gust_mph: 24,
    temperature_f: 61,
    precipitation_probability: 60,
  },
  "2025_06_NE_NO": {
    venue_name: "Caesars Superdome",
    venue_type: "indoor",
    roof_status: "closed",
    wind_mph: 0,
    wind_gust_mph: null,
    temperature_f: 70,
    precipitation_probability: 0,
  },
  "2025_06_PHI_NYG": {
    venue_name: "MetLife Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 22,
    wind_gust_mph: 35,
    temperature_f: 55,
    precipitation_probability: 30,
  },
  "2025_06_SEA_JAX": {
    venue_name: "EverBank Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 7,
    wind_gust_mph: 12,
    temperature_f: 82,
    precipitation_probability: 30,
  },
  "2025_06_SF_TB": {
    venue_name: "Raymond James Stadium",
    venue_type: "outdoor",
    roof_status: "open",
    wind_mph: 4,
    wind_gust_mph: 8,
    temperature_f: 88,
    precipitation_probability: 65,
  },
  "2025_06_TEN_LV": {
    venue_name: "Allegiant Stadium",
    venue_type: "indoor",
    roof_status: "closed",
    wind_mph: 0,
    wind_gust_mph: null,
    temperature_f: 70,
    precipitation_probability: 0,
  },
};

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

async function getSchedule(season: number, week: number): Promise<ScheduleResponse> {
  try {
    const response = await fetch(`${API_URL}/schedule/${season}/${week}`, {
      next: { revalidate: 300 },
    });
    if (!response.ok) throw new Error("Schedule API unavailable");
    return response.json();
  } catch {
    return { season, week, count: 0, games: [] };
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

function kickoffDate(value: string | null) {
  if (!value) return null;
  const hasZone = /(?:Z|[+-]\d{2}:\d{2})$/.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}

function formatPacificKickoff(value: string | null) {
  const date = kickoffDate(value);
  if (!date || Number.isNaN(date.getTime())) return "Time pending";
  const formatted = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Los_Angeles",
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
  return `${formatted} PT`;
}

type WeatherSummary =
  | "sunny"
  | "partly-cloudy"
  | "cloudy"
  | "light-rain"
  | "heavy-rain"
  | "windy";

function summarizeWeather(game: Prediction): WeatherSummary {
  const wind = game.wind_mph ?? 0;
  const rain = game.precipitation_probability ?? 0;
  if (wind >= 18) return "windy";
  if (game.weather_code !== null) {
    if ([65, 67, 82, 95, 96, 99].includes(game.weather_code)) {
      return "heavy-rain";
    }
    if ([51, 53, 55, 56, 57, 61, 63, 66, 80, 81].includes(game.weather_code)) {
      return "light-rain";
    }
    if ([3, 45, 48, 71, 73, 75, 77, 85, 86].includes(game.weather_code)) {
      return "cloudy";
    }
    if ([1, 2].includes(game.weather_code)) return "partly-cloudy";
    if (game.weather_code === 0) return "sunny";
  }
  if (rain >= 65) return "heavy-rain";
  if (rain >= 40) return "light-rain";
  if (rain >= 25) return "cloudy";
  if (rain >= 10) return "partly-cloudy";
  return "sunny";
}

function WeatherIcon({ summary }: { summary: WeatherSummary }) {
  const label = summary.replace("-", " ");
  if (summary === "sunny") {
    return (
      <span className="weather-icon sunny" title={label} aria-label={label}>
        <svg viewBox="0 0 32 32" aria-hidden="true">
          <circle cx="16" cy="16" r="6" />
          <path d="M16 3v4M16 25v4M3 16h4M25 16h4M6.8 6.8l2.8 2.8M22.4 22.4l2.8 2.8M25.2 6.8l-2.8 2.8M9.6 22.4l-2.8 2.8" />
        </svg>
      </span>
    );
  }
  if (summary === "windy") {
    return (
      <span className="weather-icon windy" title={label} aria-label={label}>
        <svg viewBox="0 0 32 32" aria-hidden="true">
          <path d="M4 11h15c4 0 4-6 0-6-2 0-3 1-3 3M4 16h21c4 0 4 6 0 6-2 0-3-1-3-3M4 21h11" />
        </svg>
      </span>
    );
  }
  const raining = summary === "light-rain" || summary === "heavy-rain";
  return (
    <span
      className={`weather-icon ${summary}`}
      title={label}
      aria-label={label}
    >
      <svg viewBox="0 0 32 32" aria-hidden="true">
        {summary === "partly-cloudy" && (
          <>
            <circle className="icon-sun" cx="11" cy="10" r="5" />
            <path className="icon-ray" d="M11 2v3M3 10h3M5.5 4.5l2 2" />
          </>
        )}
        <path
          className="icon-cloud"
          d="M7 23h17a5 5 0 0 0 0-10 8 8 0 0 0-15-2 6 6 0 0 0-2 12Z"
        />
        {raining && (
          <path
            className="icon-rain"
            d={
              summary === "heavy-rain"
                ? "M10 25l-2 4M17 25l-2 4M24 25l-2 4"
                : "M12 25l-2 4M21 25l-2 4"
            }
          />
        )}
      </svg>
    </span>
  );
}

const INTERNATIONAL_FLAGS: Record<string, { src: string; country: string }> = {
  AU: { src: "/country-flags/AU.svg", country: "Australia" },
  BR: { src: "/country-flags/BR.svg", country: "Brazil" },
  DE: { src: "/country-flags/DE.svg", country: "Germany" },
  ES: { src: "/country-flags/ES.svg", country: "Spain" },
  FR: { src: "/country-flags/FR.svg", country: "France" },
  GB: { src: "/country-flags/GB.svg", country: "United Kingdom" },
  MX: { src: "/country-flags/MX.svg", country: "Mexico" },
};

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
  const international =
    game.country_code ? INTERNATIONAL_FLAGS[game.country_code] : undefined;
  const highConfidenceSignal =
    game.moneyline_signal &&
    game.model_win_confidence !== null &&
    game.model_win_confidence >= 0.65;
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
        <span className="game-status">
          <span>{completed ? "Final" : `Week ${week}`}</span>
          {international && (
            <Image
              className="venue-flag"
              src={international.src}
              alt={`${international.country} flag`}
              title={`Game played in ${international.country}`}
              width={26}
              height={17}
            />
          )}
        </span>
        <time className="kickoff-time" dateTime={game.kickoff ?? undefined}>
          {formatPacificKickoff(game.kickoff)}
        </time>
        <span>
          {game.model_version
            ? game.model_version.replaceAll("-", " ")
            : "Official prediction pending"}
        </span>
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
                <span>Median projection</span>
                <b>{game.predicted_away_score ?? "—"}</b>
              </div>
              <div className="score-column final-score">
                <span>Final</span>
                <b>{game.actual_away_score}</b>
              </div>
            </div>
          ) : (
            <div className="score-comparison pregame-comparison">
              <div className="score-column predicted-score">
                <span>Median projection</span>
                <b>{game.predicted_away_score ?? "—"}</b>
              </div>
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
                <span>Median projection</span>
                <b>{game.predicted_home_score ?? "—"}</b>
              </div>
              <div className="score-column final-score">
                <span>Final</span>
                <b>{game.actual_home_score}</b>
              </div>
            </div>
          ) : (
            <div className="score-comparison pregame-comparison">
              <div className="score-column predicted-score">
                <span>Median projection</span>
                <b>{game.predicted_home_score ?? "—"}</b>
              </div>
            </div>
          )}
        </div>
      </div>
      {game.weather_source && (
        <div className="game-conditions">
          {game.venue_type !== "indoor" && (
            game.schedule_only ? (
              <span className="weather-icon pending" aria-label="Weather pending">
                —
              </span>
            ) : (
              <WeatherIcon summary={summarizeWeather(game)} />
            )
          )}
          <span>
            {game.roof_status === "pending"
              ? "Roof pending"
              : game.venue_type === "indoor"
                ? "Indoors"
                : game.schedule_only
                  ? "Outdoor"
                : `${Math.round(game.wind_mph ?? 0)} mph wind`}
          </span>
          {game.schedule_only && game.venue_type !== "indoor" && (
            <>
              <span>Wind —</span>
              <span>Gusts —</span>
              <span>Temp —</span>
              <span>Rain —</span>
            </>
          )}
          {game.venue_type !== "indoor" && game.wind_gust_mph !== null && (
            <span>Gusts {Math.round(game.wind_gust_mph)} mph</span>
          )}
          {game.temperature_f !== null && (
            <span>{Math.round(game.temperature_f)}°F</span>
          )}
          {game.venue_type !== "indoor" &&
            game.weather_source === "open-meteo-historical-forecast" &&
            game.precipitation_inches !== null &&
            game.precipitation_inches > 0 && (
              <span>Rain {game.precipitation_inches.toFixed(2)} in</span>
            )}
          {game.venue_type !== "indoor" &&
            game.weather_source !== "open-meteo-historical-forecast" &&
            game.precipitation_probability !== null && (
              <span>
                Rain {Math.round(game.precipitation_probability)}%
              </span>
            )}
          <span className="conditions-source">
            {game.weather_source === "demo"
              ? "Weather layout demo"
              : game.weather_source === "schedule"
                ? "Forecast pending"
              : game.weather_source === "open-meteo-historical-forecast"
                ? "Historical weather · Open-Meteo"
              : "Open-Meteo forecast"}
          </span>
        </div>
      )}
      <div className="prediction-footer">
        <div>
          <span className="eyebrow">Model pick</span>
          <strong>{game.predicted_winner ?? "—"}</strong>
        </div>
        <div>
          <span className="eyebrow">Confidence</span>
          <strong>
            {game.model_win_confidence === null
              ? "—"
              : `${(game.model_win_confidence * 100).toFixed(1)}%`}
          </strong>
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
          <span
            className={`signal-badge ${highConfidenceSignal ? "high-confidence" : ""}`}
            title="Experimental moneyline signal"
          >
            {highConfidenceSignal ? "High-confidence signal" : "Signal"} indicates{" "}
            {game.predicted_winner}{" "}
            {formatOdds(game.moneyline_signal_odds)}
          </span>
        )}
        {game.moneyline_signal && completed && (
          <span
            className={`signal-badge ${highConfidenceSignal ? "high-confidence" : ""} ${
              game.moneyline_signal_won ? "won" : "lost"
            }`}
          >
            {highConfidenceSignal ? "High-confidence signal" : "Signal"}{" "}
            {game.moneyline_signal_won ? "won" : "lost"} ·{" "}
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
  const scheduleBoard =
    selectedSeason === 2026
      ? await getSchedule(selectedSeason, selectedWeek)
      : { season: selectedSeason, week: selectedWeek, count: 0, games: [] };
  const previousWeek = Math.max(6, selectedWeek - 1);
  const nextWeek = Math.min(18, selectedWeek + 1);
  const weatherPreview =
    pregamePreview && selectedSeason === 2025 && selectedWeek === 6;
  const scheduleCards: Prediction[] = scheduleBoard.games.map((game) => ({
    ...game,
    predicted_home_score: null,
    predicted_away_score: null,
    predicted_winner: null,
    model_win_confidence: null,
    moneyline_signal: false,
    home_moneyline: null,
    away_moneyline: null,
    moneyline_signal_odds: null,
    model_version: null,
    generated_at: "",
    actual_home_score: null,
    actual_away_score: null,
    prediction_correct: null,
    moneyline_signal_won: null,
    moneyline_signal_profit: null,
    forecast_for: game.kickoff,
    weather_retrieved_at: null,
    wind_mph: null,
    wind_gust_mph: null,
    temperature_f: null,
    precipitation_probability: null,
    precipitation_inches: null,
    weather_code: null,
    weather_source: "schedule",
    schedule_only: true,
  }));
  const baseCards = board.count > 0 ? board.predictions : scheduleCards;
  const displayedPredictions = (weatherPreview
    ? baseCards.map((game) => ({
        ...game,
        ...WEATHER_PREVIEW_2025_WEEK_6[game.game_id],
        weather_source: "demo",
      }))
    : baseCards
  ).sort((a, b) => {
    const first = kickoffDate(a.kickoff)?.getTime() ?? Number.MAX_SAFE_INTEGER;
    const second = kickoffDate(b.kickoff)?.getTime() ?? Number.MAX_SAFE_INTEGER;
    return first - second || a.game_id.localeCompare(b.game_id);
  });

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
              <span>Signal rule</span>
              <strong>60% / −300</strong>
            </div>
          </div>
        </section>

        <section className="dashboard">
          <div className="section-heading">
            <div>
              <span className="overline">Weekly board</span>
              <h2>{selectedSeason} predictions</h2>
            </div>
            <div className="board-controls">
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
              <div className="week-controls">
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

          {selectedSeason === 2026 && (
            <aside className="signal-policy" aria-label="Moneyline signal policy">
              <strong>How official signals work</strong>
              <p>
                The signal refers to a moneyline bet: the selected team only
                needs to win the game outright, with no point spread, and the
                American odds determine the payout. A signal appears only when
                the model gives its predicted winner at least 60% confidence
                and that team&apos;s moneyline is −300 or better. For example,
                −275 and +110 qualify; −325 does not. Before the lock,
                Signals at 65% or higher are labeled high confidence so both
                tiers can be tracked separately. Before the lock, refreshed
                odds can make a signal appear or disappear. The
                latest eligible prediction locks one hour before that
                game&apos;s kickoff and cannot be rewritten afterward.
              </p>
            </aside>
          )}

          {weatherPreview && (
            <div className="weather-demo-notice">
              <strong>Weather preview demo</strong>
              <span>
                Sample conditions for layout review only—not historical
                forecasts or model inputs.
              </span>
            </div>
          )}

          {displayedPredictions.length > 0 ? (
            <div className="game-grid">
              {displayedPredictions.map((game) => (
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
            time, and available market price. Picks lock one hour before each
            game&apos;s kickoff, creating an auditable prospective record
            instead of rewriting history.
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
