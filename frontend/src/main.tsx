import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './theme.css';

// apply remembered theme before first paint
const t = localStorage.getItem('tl-theme') || 'dark';
document.documentElement.setAttribute('data-theme', t);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
