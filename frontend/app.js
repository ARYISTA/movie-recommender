/**
 * CineMatch — Netflix-style UI + FastAPI integration
 */

'use strict';

const API = typeof window !== 'undefined' && window.location ? window.location.origin : '';
const TMDB_IMG = 'https://image.tmdb.org/t/p/w500';

const state = {
  userId: null,
  username: null,
  selectedMood: null,
  selectedGenres: new Set(),
  searchTimer: null,
  suggestTimer: null,
};

document.addEventListener('DOMContentLoaded', () => {
  renderGenreChips();
  loadSavedUser();
  setupNavSearch();
  loadTrendingRow();
  loadTopRatedRow();
  loadBecauseRow();
  loadMoodRow();
});

function scrollToRow(rowId) {
  const el = document.getElementById(rowId);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function posterUrl(path) {
  if (!path) return '';
  const s = String(path).trim();
  if (s.startsWith('http')) return s;
  if (s.startsWith('/')) return TMDB_IMG + s;
  return TMDB_IMG + '/' + s.replace(/^\//, '');
}

function genresLine(m) {
  const g = m.genres;
  if (!g) return '';
  if (Array.isArray(g)) return g.filter(Boolean).slice(0, 2).join(' · ');
  return String(g).split('|').filter(Boolean).slice(0, 2).join(' · ');
}

// ── Auth ─────────────────────────────────────────────────────────────────────

function loadSavedUser() {
  const saved = localStorage.getItem('cinematch_user');
  if (!saved) return;
  try {
    const { userId, username } = JSON.parse(saved);
    state.userId = userId;
    state.username = username;
    updateUserLabel();
    loadHistory();
    loadProfile();
    loadBecauseRow();
    loadMoodRow();
  } catch {
    localStorage.removeItem('cinematch_user');
  }
}

async function loginOrCreate() {
  const username = document.getElementById('usernameInput').value.trim();
  if (!username) {
    showToast('Enter a username first', 'error');
    return;
  }
  if (username.length < 3) {
    showToast('Username must be at least 3 characters', 'error');
    return;
  }

  try {
    const res = await fetch(`${API}/api/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username }),
    });

    if (res.status === 409) {
      const lookup = await fetch(
        `${API}/api/users/lookup?username=${encodeURIComponent(username)}`
      );
      if (!lookup.ok) {
        showToast('Could not sign you in. Try again.', 'error');
        return;
      }
      const data = await lookup.json();
      state.userId = data.id;
      state.username = data.username;
      showToast(`Welcome back, ${username}`, 'success');
    } else if (res.ok) {
      const data = await res.json();
      state.userId = data.id;
      state.username = data.username;
      showToast(`Welcome, ${username}`, 'success');
    } else {
      showToast('Login failed. Try again.', 'error');
      return;
    }

    localStorage.setItem(
      'cinematch_user',
      JSON.stringify({ userId: state.userId, username: state.username })
    );
    updateUserLabel();
    loadHistory();
    loadProfile();
    loadBecauseRow();
    loadMoodRow();
  } catch (err) {
    showToast('Cannot reach server. Is it running?', 'error');
    console.error(err);
  }
}

function updateUserLabel() {
  const el = document.getElementById('userLabel');
  if (state.username) {
    el.textContent = state.username;
    document.getElementById('usernameInput').value = state.username;
  } else {
    el.textContent = '';
  }
}

// ── Mood & genres ────────────────────────────────────────────────────────────

function selectMood(btn) {
  document.querySelectorAll('.mood-btn').forEach((b) => b.classList.remove('active'));
  if (state.selectedMood === btn.dataset.mood) {
    state.selectedMood = null;
  } else {
    btn.classList.add('active');
    state.selectedMood = btn.dataset.mood;
  }
  loadMoodRow();
}

function renderGenreChips() {
  const container = document.getElementById('genreChips');
  const GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western',
  ];
  container.innerHTML = GENRES.map(
    (g) => `<button type="button" class="genre-chip" data-genre="${g}" onclick="toggleGenre(this)">${g}</button>`
  ).join('');
}

function toggleGenre(chip) {
  const genre = chip.dataset.genre;
  if (state.selectedGenres.has(genre)) {
    state.selectedGenres.delete(genre);
    chip.classList.remove('active');
  } else {
    state.selectedGenres.add(genre);
    chip.classList.add('active');
  }
  loadMoodRow();
}

// ── Rows ─────────────────────────────────────────────────────────────────────

function rowSkeleton(n) {
  return `<div class="skeleton-row" style="max-width:1280px;margin:0 auto">${Array(n)
    .fill(0)
    .map(() => '<div class="skeleton-tile"></div>')
    .join('')}</div>`;
}

function scrollRow(elementId, direction) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const delta = Math.min(480, Math.floor(el.clientWidth * 0.85));
  el.scrollBy({ left: direction * delta, behavior: 'smooth' });
}

async function loadTrendingRow() {
  const row = document.getElementById('rowTrending');
  row.innerHTML = rowSkeleton(8);
  try {
    const res = await fetch(`${API}/api/movies/trending?limit=20`);
    if (!res.ok) throw new Error(await res.text());
    const movies = await res.json();
    if (!movies.length) {
      row.innerHTML =
        '<p class="row-empty">No trending titles. Configure <code>TMDB_API_KEY</code> in <code>.env</code>.</p>';
      return;
    }
    // Use first trending poster as cinematic hero banner (if available).
    const firstPoster = posterUrl(movies[0]?.poster_path);
    if (firstPoster) {
      const hero = document.querySelector('.hero-inner');
      if (hero) hero.style.setProperty('--hero-bg', `url(${firstPoster})`);
    }
    row.innerHTML = movies
      .map((m) => tileRowHTML(m, { kind: m?.tmdb_id ? 'tmdb' : 'local' }))
      .join('');
  } catch (e) {
    console.error(e);
    row.innerHTML = '<p class="row-empty">Could not load trending.</p>';
  }
}

async function loadTopRatedRow() {
  const row = document.getElementById('rowTopRated');
  row.innerHTML = rowSkeleton(8);
  try {
    const res = await fetch(`${API}/api/movies/top-rated?limit=20`);
    if (!res.ok) throw new Error(await res.text());
    const movies = await res.json();
    if (!movies.length) {
      row.innerHTML =
        '<p class="row-empty">No catalog ratings yet. Seed the database and train models.</p>';
      return;
    }
    row.innerHTML = movies.map((m) => tileRowHTML(m, { kind: 'local' })).join('');
  } catch (e) {
    console.error(e);
    row.innerHTML = '<p class="row-empty">Could not load top-rated titles.</p>';
  }
}

async function loadBecauseRow() {
  const block = document.getElementById('rowBecauseBlock');
  const row = document.getElementById('rowBecause');
  if (!state.userId) {
    block.classList.add('hidden');
    return;
  }
  block.classList.remove('hidden');
  row.innerHTML = rowSkeleton(6);
  try {
    const res = await fetch(`${API}/api/recommend/${state.userId}?top_n=18`);
    if (!res.ok) throw new Error(await res.text());
    const movies = await res.json();
    if (!movies.length) {
      row.innerHTML =
        '<p class="row-empty">No picks yet — watch or rate a few movies to tune recommendations.</p>';
      return;
    }
    row.innerHTML = movies.map((m) => tileRowHTML(m, { kind: 'rec', showScore: true })).join('');
  } catch (e) {
    console.error(e);
    row.innerHTML =
      '<p class="row-empty">Recommendations unavailable. Ensure models are trained and the API is healthy.</p>';
  }
}

async function loadMoodRow() {
  const row = document.getElementById('rowMood');
  const hint = document.getElementById('rowMoodHint');

  if (!state.userId) {
    hint.textContent = 'Sign in to see mood-based picks.';
    row.innerHTML = '<p class="row-empty">Sign in to see mood-based picks.</p>';
    return;
  }
  if (!state.selectedMood) {
    hint.textContent = 'Choose a mood in the panel below.';
    row.innerHTML = '<p class="row-empty">Select a mood to populate this row.</p>';
    return;
  }

  hint.textContent = `Picks for “${state.selectedMood}” — updates when you change mood or genres.`;
  row.innerHTML = rowSkeleton(6);

  try {
    const params = new URLSearchParams({ top_n: 18, mood: state.selectedMood });
    state.selectedGenres.forEach((g) => params.append('genres', g));
    const res = await fetch(`${API}/api/recommend/${state.userId}?${params}`);
    if (!res.ok) throw new Error(await res.text());
    const movies = await res.json();
    if (!movies.length) {
      row.innerHTML = '<p class="row-empty">No mood matches yet. Try other genres or watch more films.</p>';
      return;
    }
    row.innerHTML = movies.map((m) => tileRowHTML(m, { kind: 'rec', showScore: true })).join('');
  } catch (e) {
    console.error(e);
    row.innerHTML = '<p class="row-empty">Could not load mood picks.</p>';
  }
}

function tileRowHTML(m, opts) {
  const kind = opts.kind || 'local';
  const showScore = !!opts.showScore;
  const title = esc(m.title || 'Unknown');
  const poster = posterUrl(m.poster_path);
  const rating =
    m.vote_average != null && Number(m.vote_average) > 0
      ? `⭐ ${Number(m.vote_average).toFixed(1)}`
      : '';
  const expl = m.explanation ? esc(m.explanation) : '';
  const scorePct =
    m.final_score != null && !Number.isNaN(Number(m.final_score))
      ? Math.round(Number(m.final_score) * 100)
      : null;

  const img = poster
    ? `<img class="tile-poster" src="${poster}" alt="${title}" loading="lazy" decoding="async" onerror="this.replaceWith(tileBrokenPlaceholder())" />`
    : `<div class="tile-placeholder" aria-hidden="true">🎬</div>`;

  if (kind === 'tmdb') {
    const tid = m.tmdb_id;
    const titleJson = JSON.stringify(m.title || '');
    return `
      <article class="tile">
        <div class="tile-poster-wrap" role="button" tabindex="0" onclick="openTmdbModal(${tid})" onkeydown="if(event.key==='Enter')openTmdbModal(${tid})">
          ${img}
          <div class="tile-overlay">
            <div class="tile-meta">
              <div class="tile-title">${title}</div>
              ${rating ? `<div class="tile-rating">${rating}</div>` : ''}
            </div>
            <div class="tile-actions" onclick="event.stopPropagation()">
              <button type="button" class="tile-btn tile-btn-watch" onclick="openTmdbModal(${tid})">Details</button>
              <button type="button" class="tile-btn tile-btn-rate" onclick="prefillSearchFromTitle(${titleJson})">Find in catalog</button>
            </div>
          </div>
        </div>
      </article>`;
  }

  const id = m.movie_id ?? m.id;
  return `
    <article class="tile">
      <div class="tile-poster-wrap" role="button" tabindex="0" onclick="openMovieModal(${id})" onkeydown="if(event.key==='Enter')openMovieModal(${id})">
        ${img}
        <div class="tile-overlay">
          <div class="tile-meta">
            <div class="tile-title">${title}</div>
            ${rating ? `<div class="tile-rating">${rating}</div>` : ''}
            ${expl ? `<div class="tile-expl">${expl}</div>` : ''}
            ${
              showScore && scorePct !== null
                ? `<div class="tile-score"><span style="width:${scorePct}%"></span></div>`
                : ''
            }
          </div>
          <div class="tile-actions" onclick="event.stopPropagation()">
            <button type="button" class="tile-btn tile-btn-watch" onclick="openMovieModal(${id})">Watch</button>
            <button type="button" class="tile-btn tile-btn-rate" onclick="submitInlineRating(${id}, 5)">Rate 5★</button>
          </div>
        </div>
      </div>
    </article>`;
}

function tileBrokenPlaceholder() {
  const d = document.createElement('div');
  d.className = 'tile-placeholder';
  d.textContent = '🎬';
  return d;
}

window.tileBrokenPlaceholder = tileBrokenPlaceholder;

function prefillSearchFromTitle(title) {
  const input = document.getElementById('navSearchInput');
  input.value = title || '';
  hideSuggestions();
  runFullSearch((title || '').trim());
  document.getElementById('searchSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Full recommendations (grid) ─────────────────────────────────────────────

async function getRecommendations() {
  if (!state.userId) {
    showToast('Please sign in first', 'error');
    return;
  }

  const grid = document.getElementById('movieGrid');
  const section = document.getElementById('resultsSection');
  section.classList.remove('hidden');
  grid.innerHTML = spinnerHTML();

  const params = new URLSearchParams({ top_n: 20 });
  if (state.selectedMood) params.append('mood', state.selectedMood);
  state.selectedGenres.forEach((g) => params.append('genres', g));

  try {
    const res = await fetch(`${API}/api/recommend/${state.userId}?${params}`);
    if (!res.ok) throw new Error(await res.text());
    const movies = await res.json();

    if (!movies.length) {
      grid.innerHTML = emptyStateHTML('No results', 'Try different moods or genres, or watch more titles.');
      return;
    }

    grid.innerHTML = movies.map((m) => gridCardHTML(m, true)).join('');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    loadBecauseRow();
  } catch (err) {
    grid.innerHTML = emptyStateHTML('Error', 'Could not load recommendations.');
    console.error(err);
  }
}

// ── Navbar search & suggestions ─────────────────────────────────────────────

function setupNavSearch() {
  const input = document.getElementById('navSearchInput');
  const wrap = document.getElementById('navSearchWrap');
  const btn = document.getElementById('navSearchBtn');

  input.addEventListener('input', () => {
    clearTimeout(state.suggestTimer);
    const q = input.value.trim();
    if (!q) {
      hideSuggestions();
      return;
    }
    state.suggestTimer = setTimeout(() => fetchSuggestions(q), 280);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      hideSuggestions();
      const q = input.value.trim();
      if (q) runFullSearch(q);
    }
    if (e.key === 'Escape') hideSuggestions();
  });

  btn?.addEventListener('click', () => {
    hideSuggestions();
    const q = input.value.trim();
    if (q) runFullSearch(q);
    input.focus();
  });

  document.addEventListener('click', (e) => {
    if (!wrap.contains(e.target)) hideSuggestions();
  });
}

function hideSuggestions() {
  const box = document.getElementById('searchSuggestions');
  box.classList.add('hidden');
  box.innerHTML = '';
  document.getElementById('navSearchInput').setAttribute('aria-expanded', 'false');
}

async function fetchSuggestions(q) {
  const box = document.getElementById('searchSuggestions');
  try {
    const res = await fetch(`${API}/api/movies/search?q=${encodeURIComponent(q)}&limit=8`);
    if (!res.ok) throw new Error();
    const movies = await res.json();
    if (!movies.length) {
      hideSuggestions();
      return;
    }
    box.innerHTML = movies
      .map((m) => {
        const id = m.id;
        const pt = posterUrl(m.poster_path);
        const thumb = pt
          ? `<img class="suggest-poster" src="${pt}" alt="" loading="lazy" decoding="async" />`
          : `<div class="suggest-poster" style="display:flex;align-items:center;justify-content:center;font-size:1.2rem;background:var(--surface2)">🎬</div>`;
        const sub = [genresLine(m), m.year].filter(Boolean).join(' · ');
        const titleHTML = highlightMatch(esc(m.title), q);
        const subHTML = highlightMatch(esc(sub), q);
        return `<button type="button" role="option" onclick="pickSuggestion(${id})">
          ${thumb}
          <div class="suggest-meta"><div class="suggest-title">${titleHTML}</div><div class="suggest-sub">${subHTML}</div></div>
        </button>`;
      })
      .join('');
    box.classList.remove('hidden');
    document.getElementById('navSearchInput').setAttribute('aria-expanded', 'true');
  } catch {
    hideSuggestions();
  }
}

function highlightMatch(htmlSafeText, query) {
  const q = (query || '').trim();
  if (!q) return htmlSafeText;
  const hay = htmlSafeText;
  const needle = esc(q);
  if (!needle) return hay;

  // Case-insensitive highlight without breaking HTML entities too much.
  // We operate on already-escaped strings to avoid XSS.
  const re = new RegExp(escapeRegExp(needle), 'ig');
  return hay.replace(re, (m) => `<mark>${m}</mark>`);
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function pickSuggestion(movieId) {
  hideSuggestions();
  openMovieModal(movieId);
}

async function runFullSearch(q) {
  const section = document.getElementById('searchSection');
  const grid = document.getElementById('searchGrid');
  section.classList.remove('hidden');
  grid.innerHTML = spinnerHTML();

  try {
    const res = await fetch(`${API}/api/movies/search?q=${encodeURIComponent(q)}&limit=24`);
    if (!res.ok) throw new Error(await res.text());
    const movies = await res.json();

    if (!movies.length) {
      grid.innerHTML = emptyStateHTML('No matches', 'Try another title or keyword.');
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    grid.innerHTML = movies.map((m) => gridCardHTML(m, false)).join('');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch {
    grid.innerHTML = emptyStateHTML('Search failed', 'Check your connection and try again.');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ── History & profile ────────────────────────────────────────────────────────

async function loadHistory() {
  if (!state.userId) return;
  const section = document.getElementById('historySection');
  const grid = document.getElementById('historyGrid');

  try {
    const res = await fetch(`${API}/api/users/${state.userId}/history?limit=12`);
    const data = await res.json();
    if (!data.history?.length) {
      section.classList.add('hidden');
      return;
    }

    section.classList.remove('hidden');
    grid.innerHTML = data.history
      .map((m) =>
        gridCardHTML(
          {
            movie_id: m.movie_id,
            title: m.title,
            genres: m.genres,
            poster_path: m.poster_path,
            vote_average: 0,
            explanation: `Watched · ${timeAgo(m.watched_at)}`,
          },
          false
        )
      )
      .join('');
  } catch (err) {
    console.error('History load failed', err);
  }
}

async function logWatch(movieId, title) {
  if (!state.userId) {
    showToast('Sign in to track watches', 'error');
    return;
  }

  try {
    const res = await fetch(`${API}/api/users/${state.userId}/watch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: movieId, watch_progress: 100 }),
    });
    if (!res.ok) throw new Error(await res.text());
    showToast(`Added “${title}” to history`, 'success');
    loadHistory();
    loadProfile();
    loadBecauseRow();
  } catch {
    showToast('Could not log watch', 'error');
  }
}

async function rateMovie(movieId, score) {
  if (!state.userId) {
    showToast('Sign in to rate movies', 'error');
    return;
  }

  try {
    const res = await fetch(`${API}/api/users/${state.userId}/rate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: movieId, score }),
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t);
    }
    showToast(`Rated ${score} / 5`, 'success');
    loadBecauseRow();
  } catch {
    showToast('Rating failed', 'error');
  }
}

async function loadProfile() {
  if (!state.userId) return;

  try {
    const res = await fetch(`${API}/api/users/${state.userId}/profile`);
    if (!res.ok) return;
    const profile = await res.json();

    const section = document.getElementById('profileSection');
    section.classList.remove('hidden');

    const breakdown = profile.genre_breakdown || {};
    const sortedGenres = Object.entries(breakdown)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    const maxPct = sortedGenres.length ? sortedGenres[0][1] : 1;

    document.getElementById('profileContent').innerHTML = `
      <div class="profile-grid">
        <div class="profile-stat">
          <div class="label">Movies watched</div>
          <div class="value">${profile.total_watched ?? 0}</div>
        </div>
        <div class="profile-stat">
          <div class="label">Top genre</div>
          <div class="value" style="font-size:18px">${profile.top_genres?.[0] ?? '—'}</div>
        </div>
      </div>
      ${
        sortedGenres.length
          ? `
        <div class="genre-bars" style="margin-top:20px">
          <div class="label" style="margin-bottom:10px;font-size:13px;color:var(--text-muted)">Genre breakdown</div>
          ${sortedGenres
            .map(
              ([genre, pct]) => `
            <div class="genre-bar-row">
              <span class="genre-bar-label">${esc(genre)}</span>
              <div class="genre-bar-track">
                <div class="genre-bar-fill" style="width:${Math.round((pct / maxPct) * 100)}%"></div>
              </div>
              <span class="genre-bar-pct">${Math.round(pct * 100)}%</span>
            </div>
          `
            )
            .join('')}
        </div>
      `
          : ''
      }
    `;
  } catch (err) {
    console.error('Profile load failed', err);
  }
}

// ── Modals ───────────────────────────────────────────────────────────────────

async function openMovieModal(movieId) {
  const overlay = document.getElementById('modalOverlay');
  const content = document.getElementById('modalContent');
  overlay.classList.remove('hidden');
  content.innerHTML = spinnerHTMLModal();

  try {
    const res = await fetch(`${API}/api/movies/${movieId}`);
    if (!res.ok) throw new Error();
    const m = await res.json();

    const genreTags = (Array.isArray(m.genres) ? m.genres : String(m.genres || '').split('|'))
      .filter(Boolean)
      .map((g) => `<span class="genres-tag">${esc(g)}</span>`)
      .join('');

    const castStr = m.cast ? String(m.cast).replace(/\|/g, ', ') : '';

    content.innerHTML = `
      <div class="modal-inner">
        ${
          m.poster_path
            ? `<img class="modal-poster" src="${posterUrl(m.poster_path)}" alt="${esc(m.title)}" loading="lazy" />`
            : `<div class="modal-poster tile-placeholder" style="display:flex;align-items:center;justify-content:center;font-size:3rem" aria-hidden="true">🎬</div>`
        }
        <div class="modal-details">
          <h3 id="modalTitle">${esc(m.title)} ${
      m.year
        ? `<span style="color:var(--text-muted);font-weight:500;font-size:0.85em">(${m.year})</span>`
        : ''
    }</h3>
          <div style="margin:8px 0">${genreTags}</div>
          ${m.director ? `<div style="font-size:13px;color:var(--text-muted);margin-top:6px">${esc(m.director)}</div>` : ''}
          ${castStr ? `<div style="font-size:13px;color:var(--text-muted);margin-top:4px">${esc(castStr)}</div>` : ''}
          ${
            m.vote_average
              ? `<div class="movie-rating" style="margin-top:8px">⭐ ${Number(m.vote_average).toFixed(1)} / 10</div>`
              : ''
          }
          ${m.overview ? `<p class="modal-overview">${esc(m.overview)}</p>` : ''}
          <div class="star-row" id="starRow-${movieId}">
            ${[1, 2, 3, 4, 5]
              .map((s) => `<span title="Rate ${s}" onclick="submitRating(${movieId},${s})">☆</span>`)
              .join('')}
          </div>
          <div class="modal-actions">
            <button type="button" class="btn-primary" onclick="logWatch(${movieId}, ${JSON.stringify(m.title)})">Mark as watched</button>
            <button type="button" class="btn-secondary" onclick="document.getElementById('modalOverlay').classList.add('hidden')">Close</button>
          </div>
        </div>
      </div>
    `;
  } catch {
    content.innerHTML = `<p class="modal-overview">Could not load movie details.</p>`;
  }
}

async function openTmdbModal(tmdbId) {
  const overlay = document.getElementById('modalOverlay');
  const content = document.getElementById('modalContent');
  overlay.classList.remove('hidden');
  content.innerHTML = spinnerHTMLModal();

  try {
    const res = await fetch(`${API}/api/tmdb/movie/${tmdbId}`);
    if (res.status === 503) {
      content.innerHTML = `<p class="modal-overview">TMDB is not configured. Add <code>TMDB_API_KEY</code> to your <code>.env</code> file.</p>`;
      return;
    }
    if (!res.ok) throw new Error();
    const m = await res.json();

    const genreTags = (m.genres || [])
      .filter(Boolean)
      .map((g) => `<span class="genres-tag">${esc(g)}</span>`)
      .join('');
    const castStr = m.cast ? String(m.cast).replace(/\|/g, ', ') : '';

    const tmdbUrl = `https://www.themoviedb.org/movie/${tmdbId}`;

    content.innerHTML = `
      <div class="modal-inner">
        ${
          m.poster_path
            ? `<img class="modal-poster" src="${posterUrl(m.poster_path)}" alt="${esc(m.title)}" loading="lazy" />`
            : `<div class="modal-poster tile-placeholder" style="display:flex;align-items:center;justify-content:center;font-size:3rem">🎬</div>`
        }
        <div class="modal-details">
          <h3>${esc(m.title)} ${
      m.year
        ? `<span style="color:var(--text-muted);font-weight:500;font-size:0.85em">(${m.year})</span>`
        : ''
    }</h3>
          <div style="margin:8px 0">${genreTags}</div>
          ${m.director ? `<div style="font-size:13px;color:var(--text-muted);margin-top:6px">${esc(m.director)}</div>` : ''}
          ${castStr ? `<div style="font-size:13px;color:var(--text-muted);margin-top:4px">${esc(castStr)}</div>` : ''}
          ${
            m.vote_average
              ? `<div class="movie-rating" style="margin-top:8px">⭐ ${Number(m.vote_average).toFixed(1)} / 10</div>`
              : ''
          }
          ${m.overview ? `<p class="modal-overview">${esc(m.overview)}</p>` : ''}
          <p class="modal-overview" style="font-size:0.85rem">Watches and ratings apply to titles in our catalog. Use search to find a match.</p>
          <div class="modal-actions">
            <button type="button" class="btn-primary" onclick="prefillSearchFromTitle(${JSON.stringify(m.title || '')})">Search in catalog</button>
            <a class="modal-link btn-secondary" href="${tmdbUrl}" target="_blank" rel="noopener noreferrer">Open on TMDB</a>
          </div>
        </div>
      </div>
    `;
  } catch {
    content.innerHTML = `<p class="modal-overview">Could not load TMDB details.</p>`;
  }
}

async function submitRating(movieId, score) {
  const row = document.getElementById(`starRow-${movieId}`);
    if (row) {
    [...row.children].forEach((s, i) => {
      s.textContent = i < score ? '⭐' : '☆';
    });
  }
  await rateMovie(movieId, score);
}

function closeModal(e) {
  if (e.target.id === 'modalOverlay') {
    document.getElementById('modalOverlay').classList.add('hidden');
  }
}

// ── Grid cards ───────────────────────────────────────────────────────────────

function gridCardHTML(m, showScore) {
  const id = m.movie_id ?? m.id;
  const title = esc(m.title ?? 'Unknown');
  const meta = [genresLine(m), m.year].filter(Boolean).join(' · ');
  const rating =
    m.vote_average != null && Number(m.vote_average) > 0
      ? `⭐ ${Number(m.vote_average).toFixed(1)}`
      : '';
  const score =
    m.final_score != null && !Number.isNaN(Number(m.final_score))
      ? Math.round(Number(m.final_score) * 100)
      : null;
  const expl = m.explanation ? esc(m.explanation) : '';

  const poster = posterUrl(m.poster_path);
  const posterBlock = poster
    ? `<img class="movie-poster" src="${poster}" alt="${title}" loading="lazy" decoding="async" onerror="this.replaceWith(tileBrokenPlaceholder())" />`
    : `<div class="poster-placeholder">🎬</div>`;

  return `
    <div class="movie-card" onclick="openMovieModal(${id})">
      <div class="poster-wrap">
        ${posterBlock}
        <div class="movie-card-overlay" onclick="event.stopPropagation()">
          <div class="movie-title">${title}</div>
          ${rating ? `<div class="movie-rating">${rating}</div>` : ''}
          <div class="card-actions">
            <button type="button" class="btn-sm btn-watch" onclick="openMovieModal(${id})">Watch</button>
            <button type="button" class="btn-sm btn-like" onclick="submitInlineRating(${id}, 5)">Rate</button>
          </div>
        </div>
      </div>
      <div class="movie-info">
        <div class="movie-title">${title}</div>
        <div class="movie-meta">${esc(meta)}</div>
        ${rating ? `<div class="movie-rating">${rating}</div>` : ''}
        ${expl ? `<div class="movie-explanation">${expl}</div>` : ''}
        ${
          showScore && score !== null
            ? `<div class="score-bar-wrap"><div class="score-bar" style="width:${score}%"></div></div>`
            : ''
        }
      </div>
    </div>`;
}

async function submitInlineRating(movieId, score) {
  await rateMovie(movieId, score);
}

function spinnerHTML() {
  return `<div class="spinner-wrap"><div class="spinner"></div></div>`;
}

function spinnerHTMLModal() {
  return `<div class="spinner-wrap" style="padding:32px"><div class="spinner"></div></div>`;
}

function emptyStateHTML(title, msg) {
  return `<div class="empty-state"><div class="icon" aria-hidden="true">🎬</div><p><strong>${esc(title)}</strong> — ${esc(msg)}</p></div>`;
}

function showToast(msg, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast ${type}`;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 3200);
}

function esc(str) {
  const d = document.createElement('div');
  d.textContent = str == null ? '' : String(str);
  return d.innerHTML;
}

function escapeAttr(str) {
  return String(str).replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function timeAgo(isoStr) {
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(mins / 60);
  const days = Math.floor(hours / 24);
  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (mins > 0) return `${mins}m ago`;
  return 'just now';
}

// Expose for inline handlers in HTML
window.loginOrCreate = loginOrCreate;
window.selectMood = selectMood;
window.toggleGenre = toggleGenre;
window.getRecommendations = getRecommendations;
window.scrollRow = scrollRow;
window.openMovieModal = openMovieModal;
window.openTmdbModal = openTmdbModal;
window.logWatch = logWatch;
window.submitRating = submitRating;
window.submitInlineRating = submitInlineRating;
window.closeModal = closeModal;
window.pickSuggestion = pickSuggestion;
window.prefillSearchFromTitle = prefillSearchFromTitle;
window.scrollToRow = scrollToRow;
