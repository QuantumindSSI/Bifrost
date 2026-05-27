import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, HeatMap, BarChart, Bar } from 'recharts';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [demoData, setDemoData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runDemo = async (type) => {
    setLoading(true);
    try {
      const response = await axios.get(`/demo/${type}`);
      setDemoData(response.data);
      setActiveTab('demo');
    } catch (err) {
      setError(`Demo failed: ${err.message}`);
    }
    setLoading(false);
  };

  const Dashboard = () => (
    <div className="dashboard">
      <h2>🎵 FBC Dashboard</h2>
      
      <div className="actions">
        <h3>Interactive Demos</h3>
        <button onClick={() => runDemo('harmonic')} disabled={loading}>
          🎼 Harmonic Binding (440Hz ↔ 880Hz)
        </button>
        <button onClick={() => runDemo('coherence')} disabled={loading}>
          🌊 Phase Coherence
        </button>
        <button onClick={() => runDemo('multimodal')} disabled={loading}>
          🔄 Multimodal Pipeline
        </button>
      </div>

      {loading && <div className="loading">Running demo...</div>}
      {error && <div className="error">{error}</div>}
    </div>
  );

  const DemoView = () => {
    if (!demoData) return <div>No demo data</div>;

    const { demo_type, data, visualizations } = demoData;

    return (
      <div className="demo-view">
        <button className="back-btn" onClick={() => setActiveTab('dashboard')}>
          ← Back to Dashboard
        </button>
        
        <h2>Demo: {demo_type.replace('_', ' ').toUpperCase()}</h2>

        {demo_type === 'harmonic_binding' && (
          <div className="harmonic-demo">
            <div className="info">
              <h4>Frequencies: {data.frequencies.join(', ')} Hz</h4>
              <p>Harmonic bins detected: {data.harmonic_bins.length}</p>
              <p>Attention std: {data.attention_std.toFixed(4)} (non-uniform = structure)</p>
            </div>

            <div className="chart">
              <h4>Spectral Amplitude</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data.amplitude.map((v, i) => ({ x: i, value: v }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart">
              <h4>Attention Matrix (Harmonic Relationships)</h4>
              <div className="heatmap">
                {data.attention_matrix.map((row, i) => (
                  <div key={i} className="heatmap-row">
                    {row.map((val, j) => (
                      <div
                        key={j}
                        className="heatmap-cell"
                        style={{
                          backgroundColor: `rgba(136, 132, 216, ${Math.min(val * 10, 1)})`,
                          width: '3px',
                          height: '3px',
                        }}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {demo_type === 'phase_coherence' && (
          <div className="coherence-demo">
            <div className="comparison">
              <div className="metric-card">
                <h4>Coherent Phase</h4>
                <div className="metric-value">{data.coherent.smoothness.toFixed(1)}</div>
                <p>Smoothness (high = good)</p>
              </div>
              <div className="metric-card">
                <h4>Random Phase</h4>
                <div className="metric-value">{data.random.smoothness.toFixed(1)}</div>
                <p>Smoothness (low = chaotic)</p>
              </div>
              <div className="metric-card highlight">
                <h4>Improvement</h4>
                <div className="metric-value">{data.improvement_ratio.toFixed(1)}x</div>
                <p>Complex SSM advantage</p>
              </div>
            </div>

            <div className="chart">
              <h4>Phase Evolution (Coherent vs Random)</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data.coherent.phase.map((row, t) => ({
                  t,
                  coherent: row[0],
                  random: data.random.phase[t][0],
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="coherent" stroke="#82ca9d" name="Coherent" />
                  <Line type="monotone" dataKey="random" stroke="#ff7300" name="Random" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {demo_type === 'multimodal' && (
          <div className="multimodal-demo">
            <table className="modalities-table">
              <thead>
                <tr>
                  <th>Modality</th>
                  <th>Input Shape</th>
                  <th>Output Shape</th>
                  <th>SSM Type</th>
                </tr>
              </thead>
              <tbody>
                {data.modalities.map((mod, i) => (
                  <tr key={i}>
                    <td>{mod.modality}</td>
                    <td>{mod.input_shape.join(' × ')}</td>
                    <td>{mod.output_shape.join(' × ')}</td>
                    <td>{mod.ssm_type.split(' ')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="note">✅ All modalities use complex SSM with phase coherence learning</p>
          </div>
        )}

        <div className="visualizations">
          <h4>Available Visualizations</h4>
          {visualizations.map((viz, i) => (
            <span key={i} className="viz-tag">{viz}</span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 Frequency-Based Cognition</h1>
        <p>Complex SSM with Phase Coherence Learning</p>
      </header>

      <main>
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'demo' && <DemoView />}
      </main>

      <footer>
        <p>FBC v0.1.0 | ComplexSpectralDecomposer | Harmonic Binding | <a href="/docs">API Docs</a></p>
      </footer>
    </div>
  );
}

export default App;
