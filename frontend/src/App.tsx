import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import TeacherDemo from './pages/TeacherDemo';
import Compare from './pages/Compare';
import Batch from './pages/Batch';
import History from './pages/History';
import HowItWorks from './pages/HowItWorks';
import Learn from './pages/Learn';
import Privacy from './pages/Privacy';
import Settings from './pages/Settings';
import Research from './pages/Research';
import Benchmark from './pages/Benchmark';
import AiRealLab from './pages/AiRealLab';
import Validation from './pages/Validation';
import Experiments from './pages/Experiments';
import Datasets from './pages/Datasets';
import Failures from './pages/Failures';
import Generalization from './pages/Generalization';

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/teacher-demo" element={<TeacherDemo />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/history" element={<History />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/learn" element={<Learn />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/research/:id?" element={<Research />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/lab" element={<AiRealLab />} />
          <Route path="/validation" element={<Validation />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/failures" element={<Failures />} />
          <Route path="/generalization" element={<Generalization />} />
          <Route path="*" element={
            <div className="shell" style={{ padding: '60px 22px', textAlign: 'center' }}>
              <h1>404</h1>
              <p className="dim">This view does not exist.</p>
            </div>
          } />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
