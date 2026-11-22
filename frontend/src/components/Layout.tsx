import { NavLink, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/analyze', label: 'Analyze' },
  { to: '/teacher-demo', label: 'Teacher Demo' },
  { to: '/compare', label: 'Compare' },
  { to: '/batch', label: 'Batch' },
  { to: '/history', label: 'History' },
  { to: '/how-it-works', label: 'How It Works' },
  { to: '/learn', label: 'Learn' },
  { to: '/benchmark', label: 'Benchmark' },
  { to: '/validation', label: 'Validation' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/datasets', label: 'Datasets' },
  { to: '/failures', label: 'Failures' },
  { to: '/generalization', label: 'Generalization' },
  { to: '/lab', label: 'AI vs Real Lab' },
  { to: '/privacy', label: 'Privacy' },
  { to: '/settings', label: 'Settings' },
];

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="#0b0f17" stroke="rgba(148,163,184,0.25)" />
      <circle cx="16" cy="16" r="9" fill="none" stroke="#38bdf8" strokeWidth="2.5" />
      <circle cx="16" cy="16" r="3.5" fill="#38bdf8" />
      <path d="M16 2v5M16 25v5M2 16h5M25 16h5" stroke="#38bdf8" strokeWidth="2" />
    </svg>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState(() => localStorage.getItem('tl-theme') || 'dark');
  const loc = useLocation();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tl-theme', theme);
  }, [theme]);

  return (
    <>
      <nav className="nav" aria-label="Main navigation">
        <div className="shell nav-inner">
          <NavLink to="/" className="brand" aria-label="Truth Lenz AI home">
            <Logo />
            <span>TRUTH LENZ<small>AI FORENSICS</small></span>
          </NavLink>
          <div className="nav-links" role="list">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} className={({ isActive }) => (isActive ? 'active' : '')} role="listitem">
                {l.label}
              </NavLink>
            ))}
          </div>
          <button
            className="icon-btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? '☾' : '☀'}
          </button>
        </div>
      </nav>
      <main key={loc.pathname}>
        {children}
      </main>
      <footer className="footer">
        <div className="shell">
          <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
            <div>
              <strong style={{ color: 'var(--text)' }}>TRUTH LENZ AI</strong>
              <div>See beyond the pixels. Evidence behind every image.</div>
            </div>
            <div style={{ maxWidth: 520 }}>
              AI image detection is probabilistic. No detector can guarantee whether an image is
              AI-generated or authentic. Do not use Truth Lenz AI as the sole basis for legal,
              academic, employment, financial or other high-stakes decisions.
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
