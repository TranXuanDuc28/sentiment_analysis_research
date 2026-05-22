import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import axios from 'axios';
import { Sparkles, History, Send, ShieldCheck, ShieldAlert, ShieldMinus, Loader2, Globe, MessageSquare, Download, FileText, Split, Activity, Search, Info, GitMerge } from 'lucide-react';
import { analyzeSentiment, fetchHistory, analyzeUrl, analyzeCompare } from './store/sentimentSlice';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const SentimentChart = ({ positive, neutral, negative }) => {
  const total = positive + neutral + negative || 1;
  const p_p = (positive / total) * 100;
  const p_n = (neutral / total) * 100;
  const p_neg = (negative / total) * 100;

  // SVG Pie Chart math
  let cumulativePercent = 0;
  function getCoordinatesForPercent(percent) {
    const x = Math.cos(2 * Math.PI * percent);
    const y = Math.sin(2 * Math.PI * percent);
    return [x, y];
  }

  const slices = [
    { percent: p_p / 100, color: '#10b981' }, // Emerald-500
    { percent: p_n / 100, color: '#f59e0b' }, // Amber-500
    { percent: p_neg / 100, color: '#ef4444' } // Red-500
  ];

  let startPercent = 0;
  return (
    <div className="d-flex justify-content-center my-4">
      <svg viewBox="-1 -1 2 2" style={{ transform: 'rotate(-90deg)', width: '120px', height: '120px' }}>
        {slices.map((slice, index) => {
          if (slice.percent === 0) return null;
          const [startX, startY] = getCoordinatesForPercent(startPercent);
          startPercent += slice.percent;
          const [endX, endY] = getCoordinatesForPercent(startPercent);
          const largeArcFlag = slice.percent > 0.5 ? 1 : 0;
          const pathData = `M ${startX} ${startY} A 1 1 0 ${largeArcFlag} 1 ${endX} ${endY} L 0 0`;
          return <path key={index} d={pathData} fill={slice.color} />;
        })}
      </svg>
    </div>
  );
};

const AspectChart = ({ aspects }) => {
  if (!aspects || aspects.length === 0) return null;

  const categories = {};
  aspects.forEach(a => {
    if (!categories[a.aspect]) categories[a.aspect] = { pos: 0, neg: 0, neu: 0 };
    if (a.sentiment === 'Positive') categories[a.aspect].pos++;
    else if (a.sentiment === 'Negative') categories[a.aspect].neg++;
    else categories[a.aspect].neu++;
  });

  return (
    <div className="mt-4 p-3 rounded-4 bg-black bg-opacity-25 border border-white border-opacity-10">
      <h6 className="small fw-bold text-secondary mb-3 text-uppercase">Aspect Comparison</h6>
      {Object.entries(categories).map(([name, counts], idx) => {
        const total = counts.pos + counts.neg + counts.neu;
        return (
          <div key={idx} className="mb-3">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-medium text-white">{name}</span>
              <span className="opacity-50">{total} Mentions</span>
            </div>
            <div className="progress" style={{ height: '8px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
              <div className="progress-bar bg-success" style={{ width: `${(counts.pos / total) * 100}%` }}></div>
              <div className="progress-bar bg-warning" style={{ width: `${(counts.neu / total) * 100}%` }}></div>
              <div className="progress-bar bg-danger" style={{ width: `${(counts.neg / total) * 100}%` }}></div>
            </div>
          </div>
        );
      })}
      <div className="d-flex gap-3 mt-2 justify-content-center">
        <div className="small opacity-50"><span className="badge bg-success p-1 me-1">&nbsp;</span>Pos</div>
        <div className="small opacity-50"><span className="badge bg-warning p-1 me-1">&nbsp;</span>Neu</div>
        <div className="small opacity-50"><span className="badge bg-danger p-1 me-1">&nbsp;</span>Neg</div>
      </div>
    </div>
  );
};

function App() {
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [url2, setUrl2] = useState('');
  const [mode, setMode] = useState('text'); // 'text' | 'url' | 'compare' | 'domain'
  const [domainData, setDomainData] = useState(null);
  const [analyzingDomain, setAnalyzingDomain] = useState(false);
  const reportRef = React.useRef(null);

  const downloadCSV = () => {
    if (!urlResults) return;
    let csvContent = "\uFEFFText,Sentiment,Confidence,Aspects\n";
    urlResults.forEach(res => {
      const aspectsStr = (res.aspects || []).map(a => `${a.aspect}(${a.sentiment})`).join("; ");
      const row = `"${res.text.replace(/"/g, '""')}","${res.sentiment}",${res.confidence},"${aspectsStr}"`;
      csvContent += row + "\n";
    });
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `sentiment_report_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadPDF = async () => {
    if (!reportRef.current) return;
    const canvas = await html2canvas(reportRef.current, {
      backgroundColor: '#0f172a',
      scale: 2,
    });
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const imgProps = pdf.getImageProperties(imgData);
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(`sentiment_report_${new Date().getTime()}.pdf`);
  };

  const dispatch = useDispatch();
  const { currentResult, urlResults, comparisonResults, status, history, error } = useSelector((state) => state.sentiment);

  useEffect(() => {
    dispatch(fetchHistory());

    // Connect to WebSocket for real-time updates
    const socket = new WebSocket('ws://localhost:8000/ws');

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'NEW_ANALYSIS') {
        dispatch(fetchHistory());
      }
    };

    return () => socket.close();
  }, [dispatch]);

  const executeAnalysis = () => {
    const trimmedText = text.trim();
    console.log("Executing analysis in mode:", mode, "with text:", trimmedText);
    
    if (!trimmedText && (mode === 'text' || mode === 'domain')) {
      alert("Please enter some text first!");
      return;
    }
    
    if (mode === 'domain') {
      fetchDomainAnalysis(trimmedText);
    } else if (mode === 'text') {
      dispatch(analyzeSentiment(trimmedText)).then((res) => {
        if (res.payload) fetchDomainAnalysis(trimmedText);
      });
    } else if (mode === 'url' && url.trim()) {
      dispatch(analyzeUrl(url));
    } else if (mode === 'compare' && url.trim() && url2.trim()) {
      dispatch(analyzeCompare([url, url2]));
    }
  };

  const fetchDomainAnalysis = async (inputText) => {
    setAnalyzingDomain(true);
    try {
      const response = await axios.post('http://localhost:8000/api/domain-analysis', { text: inputText });
      setDomainData(response.data);
    } catch (err) {
      console.error("Domain analysis failed", err);
    } finally {
      setAnalyzingDomain(false);
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive': return <ShieldCheck className="text-emerald-400" size={48} />;
      case 'negative': return <ShieldAlert className="text-red-400" size={48} />;
      default: return <ShieldMinus className="text-amber-400" size={48} />;
    }
  };

  const getConfidenceColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive': return '#4ade80';
      case 'negative': return '#f87171';
      default: return '#fbbf24';
    }
  };

  return (
    <div className="container py-5">
      <header className="text-center mb-5">
        <h1 className="display-3 fw-bold mb-3">Sentiment AI</h1>
        <p className="lead text-secondary">Cross-Lingual Analysis for English & Vietnamese</p>
      </header>

      <main>
        <div className="row mb-4">
          <div className="col-12">
            <div className="nav nav-pills justify-content-center gap-3">
              <button
                className={`nav-link px-4 py-2 border-0 glass-card ${mode === 'text' ? 'active' : ''}`}
                onClick={() => setMode('text')}
                style={{ background: mode === 'text' ? 'var(--accent-primary)' : 'var(--glass-bg)' }}
              >
                <MessageSquare className="me-2" size={18} /> Text Input
              </button>
              <button
                className={`nav-link px-4 py-2 border-0 glass-card ${mode === 'url' ? 'active' : ''}`}
                onClick={() => setMode('url')}
                style={{ background: mode === 'url' ? 'var(--accent-primary)' : 'var(--glass-bg)' }}
              >
                <Globe className="me-2" size={18} /> URL Analysis
              </button>
              <button
                className={`nav-link px-4 py-2 border-0 glass-card ${mode === 'domain' ? 'active' : ''}`}
                onClick={() => setMode('domain')}
                style={{ background: mode === 'domain' ? 'var(--accent-primary)' : 'var(--glass-bg)' }}
              >
                <Activity className="me-2" size={18} /> Domain Insights
              </button>
            </div>
          </div>
        </div>

        <div className="row g-4 mb-5">
          <div className="col-lg-8">
            <div className="input-area glass-card h-100">
              {(mode === 'text' || mode === 'domain') && (
                <textarea
                  className="form-control bg-transparent text-white border-0 shadow-none"
                  placeholder={mode === 'domain' ? "Enter text to analyze domain shift..." : "Enter text to analyze sentiment..."}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  style={{ height: '200px', fontSize: '1.1rem' }}
                />
              )}

              {mode === 'url' && (
                <div className="py-4 px-2">
                  <label className="form-label text-secondary small fw-bold">PASTE URL TO ANALYZE CONTENT</label>
                  <input
                    type="url"
                    className="form-control bg-dark bg-opacity-25 text-white border-secondary border-opacity-25 py-3 px-4 rounded-4"
                    placeholder="https://example.com/article"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                  <p className="small text-secondary mt-3">
                    <Sparkles size={14} className="me-1" />
                    AI will automatically extract and analyze up to 10 main content segments from this URL.
                  </p>
                </div>
              )}

              {mode === 'compare' && (
                <div className="py-4 px-2">
                  <label className="form-label text-secondary small fw-bold">COMPARE TWO SOURCES</label>
                  <div className="d-flex flex-column gap-3">
                    <div className="position-relative">
                      <span className="position-absolute top-50 start-0 translate-middle-y ps-3 text-primary small fw-bold">URL 1</span>
                      <input
                        type="url"
                        className="form-control bg-dark bg-opacity-25 text-white border-secondary border-opacity-25 py-3 ps-5 pe-4 rounded-4"
                        placeholder="First URL to compare..."
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                      />
                    </div>
                    <div className="position-relative">
                      <span className="position-absolute top-50 start-0 translate-middle-y ps-3 text-info small fw-bold">URL 2</span>
                      <input
                        type="url"
                        className="form-control bg-dark bg-opacity-25 text-white border-secondary border-opacity-25 py-3 ps-5 pe-4 rounded-4"
                        placeholder="Second URL to compare..."
                        value={url2}
                        onChange={(e) => setUrl2(e.target.value)}
                      />
                    </div>
                  </div>
                  <p className="small text-secondary mt-3">
                    <Split size={14} className="me-1" /> Compare sentiment trends between two different sources side-by-side.
                  </p>
                </div>
              )}
              {mode === 'domain' && (
                <div className="px-2 py-2 mb-3">
                  <p className="small text-secondary mb-0 text-start">
                    <Activity size={14} className="me-1 text-primary" />
                    AI will visualize how this text deviates from the source training distribution.
                  </p>
                </div>
              )}
              <button
                className="btn btn-primary w-100 mt-3 py-3 rounded-4 shadow-sm border-0 d-flex align-items-center justify-content-center gap-2"
                onClick={executeAnalysis}
                style={{ background: 'linear-gradient(45deg, #3b82f6, #06b6d4)', fontWeight: 'bold' }}
              >
                {status === 'loading' || analyzingDomain ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <>Analyze Sentiment <Send size={18} /></>
                )}
              </button>
              {error && <div className="alert alert-danger mt-3 py-2 border-0 bg-opacity-10 text-danger">{error}</div>}
            </div>
          </div>

          <div className="col-lg-4">
            <div className="result-card glass-card h-100 d-flex flex-column align-items-center justify-content-center text-center" ref={reportRef}>
              {status === 'loading' ? (
                <div className="loader"></div>
              ) : mode === 'url' && urlResults ? (
                <div className="w-100 h-100 d-flex flex-column">
                  <div className="d-flex align-items-center justify-content-between mb-3">
                    <div className="d-flex align-items-center">
                      <Globe className="text-primary me-2" size={20} />
                      <h3 className="h6 mb-0 fw-bold">URL Sentiment Overview</h3>
                    </div>
                    <div className="d-flex gap-2">
                      <button
                        onClick={downloadCSV}
                        className="btn btn-sm btn-outline-primary border-0 rounded-circle p-1"
                        title="Download CSV Report"
                      >
                        <Download size={18} />
                      </button>
                      <button
                        onClick={downloadPDF}
                        className="btn btn-sm btn-outline-info border-0 rounded-circle p-1"
                        title="Download PDF Report"
                      >
                        <FileText size={18} />
                      </button>
                    </div>
                  </div>

                  {(() => {
                    const pos = urlResults.filter(r => r.sentiment === 'Positive').length;
                    const neu = urlResults.filter(r => r.sentiment === 'Neutral').length;
                    const neg = urlResults.filter(r => r.sentiment === 'Negative').length;
                    const total = urlResults.length;

                    return (
                      <div className="flex-grow-1 overflow-auto pe-2 custom-scrollbar">
                        <div className="bg-black bg-opacity-25 rounded-4 p-3 mb-3 border border-white border-opacity-10">
                          <div className="row align-items-center g-0">
                            <div className="col-5">
                              <SentimentChart positive={pos} neutral={neu} negative={neg} />
                            </div>
                            <div className="col-7 ps-3 text-start">
                              <div className="mb-2 d-flex justify-content-between">
                                <span className="small opacity-50">Positive</span>
                                <span className="fw-bold text-success">{pos}</span>
                              </div>
                              <div className="mb-2 d-flex justify-content-between">
                                <span className="small opacity-50">Neutral</span>
                                <span className="fw-bold text-warning">{neu}</span>
                              </div>
                              <div className="d-flex justify-content-between">
                                <span className="small opacity-50">Negative</span>
                                <span className="fw-bold text-danger">{neg}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Aspect Dashboard for URL */}
                        <AspectChart aspects={urlResults.flatMap(r => r.aspects || [])} />

                        <div className="mt-4">
                          <p className="small text-secondary mb-2 text-start fw-bold text-uppercase" style={{ fontSize: '0.65rem' }}>Detailed Segments ({total})</p>
                          <div className="url-results-list">
                            {urlResults.map((res, idx) => (
                              <div key={idx} className="p-3 mb-2 rounded-4 bg-black bg-opacity-25 border-start border-4" style={{ borderColor: getConfidenceColor(res.sentiment) }}>
                                <p className="small text-white mb-2 opacity-90 lh-sm">{res.text}</p>
                                <div className="d-flex justify-content-between align-items-center">
                                  <span className={`sentiment-badge py-1 px-2 ${res.sentiment.toLowerCase()}`} style={{ fontSize: '0.6rem' }}>
                                    {res.sentiment}
                                  </span>
                                  <span className="small opacity-50" style={{ fontSize: '0.65rem' }}>
                                    {(res.confidence * 100).toFixed(0)}% Match
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              ) : mode === 'compare' && comparisonResults ? (
                <div className="w-100 h-100 d-flex flex-column">
                  <h3 className="h6 mb-3 fw-bold text-start">Side-by-Side Comparison</h3>
                  <div className="row g-2 flex-grow-1 overflow-auto custom-scrollbar">
                    {comparisonResults.map((item, idx) => {
                      const pos = item.results.filter(r => r.sentiment === 'Positive').length;
                      const neu = item.results.filter(r => r.sentiment === 'Neutral').length;
                      const neg = item.results.filter(r => r.sentiment === 'Negative').length;
                      return (
                        <div key={idx} className="col-12 mb-3">
                          <div className="p-2 rounded-4 bg-black bg-opacity-25 border border-white border-opacity-10 text-start">
                            <p className="small text-truncate text-primary mb-1">{item.url}</p>
                            <div className="d-flex align-items-center">
                              <div style={{ width: '60px' }}><SentimentChart positive={pos} neutral={neu} negative={neg} /></div>
                              <div className="ms-2 flex-grow-1">
                                <div className="progress" style={{ height: '6px', borderRadius: '3px' }}>
                                  <div className="progress-bar bg-success" style={{ width: `${(pos / (pos + neu + neg || 1)) * 100}%` }}></div>
                                  <div className="progress-bar bg-warning" style={{ width: `${(neu / (pos + neu + neg || 1)) * 100}%` }}></div>
                                  <div className="progress-bar bg-danger" style={{ width: `${(neg / (pos + neu + neg || 1)) * 100}%` }}></div>
                                </div>
                                <div className="d-flex justify-content-between mt-1" style={{ fontSize: '0.6rem' }}>
                                  <span>Pos: {pos}</span>
                                  <span>Neg: {neg}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <button onClick={downloadPDF} className="btn btn-info btn-sm mt-2 rounded-pill"><FileText size={14} className="me-1" /> Export Comparison PDF</button>
                </div>
              ) : mode === 'domain' && domainData ? (
                <div className="w-100 h-100 d-flex flex-column text-start">
                  <div className="d-flex align-items-center gap-2 mb-4">
                    <Activity size={24} className="text-primary" />
                    <h3 className="h5 mb-0 fw-bold">Domain Shift Analysis</h3>
                  </div>

                  <div className="bg-black bg-opacity-25 rounded-4 p-3 mb-4 border border-white border-opacity-10">
                    <div className="d-flex justify-content-between align-items-center mb-2">
                      <span className="small text-secondary">Similarity to Source</span>
                      <span className="fw-bold text-primary">{(domainData.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress" style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
                      <div className="progress-bar bg-primary" style={{ width: `${domainData.similarity_score * 100}%` }}></div>
                    </div>
                  </div>

                  <div className="flex-grow-1 position-relative mb-4" style={{ height: '300px', background: 'rgba(0,0,0,0.2)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <svg width="100%" height="100%" viewBox="-10 -10 20 20" preserveAspectRatio="xMidYMid meet">
                      {/* Grid lines */}
                      <line x1="-10" y1="0" x2="10" y2="0" stroke="rgba(255,255,255,0.05)" strokeWidth="0.05" />
                      <line x1="0" y1="-10" x2="0" y2="10" stroke="rgba(255,255,255,0.05)" strokeWidth="0.05" />

                      {/* Source Cloud */}
                      {domainData.source_points.map((p, i) => (
                        <circle key={i} cx={p[0]} cy={p[1]} r="0.2" fill="#3b82f6" opacity="0.3" />
                      ))}

                      {/* Target Point */}
                      <g className="animate-pulse">
                        <circle cx={domainData.target_point[0]} cy={domainData.target_point[1]} r="0.6" fill="#ef4444" stroke="white" strokeWidth="0.1" />
                        <circle cx={domainData.target_point[0]} cy={domainData.target_point[1]} r="1.5" fill="#ef4444" opacity="0.15" />
                      </g>
                    </svg>

                    <div className="position-absolute bottom-0 start-0 p-3 w-100 d-flex justify-content-between small opacity-50">
                      <span>Source Distribution</span>
                      <span>Target Mapping</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-4 border border-info border-opacity-25 bg-info bg-opacity-10 mb-3">
                    <div className="d-flex align-items-start gap-2">
                      <Info size={16} className="text-info mt-1" />
                      <div>
                        <div className="fw-bold small text-info mb-1">Observation</div>
                        <p className="small mb-0 text-white opacity-75">{domainData.status}</p>
                      </div>
                    </div>
                  </div>

                  <div className="row g-2">
                    <div className="col-6">
                      <div className="p-2 rounded bg-black bg-opacity-25 border border-white border-opacity-5 text-center">
                        <div className="small text-secondary mb-1" style={{ fontSize: '0.65rem' }}>SHIFT</div>
                        <div className="fw-bold">{domainData.shift_magnitude.toFixed(3)}</div>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="p-2 rounded bg-black bg-opacity-25 border border-white border-opacity-5 text-center">
                        <div className="small text-secondary mb-1" style={{ fontSize: '0.65rem' }}>RELIABILITY</div>
                        <div className="fw-bold text-success">{(domainData.similarity_score * 100).toFixed(0)}%</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : currentResult ? (
                <>
                  {getSentimentIcon(currentResult.sentiment)}
                  <div className={`sentiment-badge mt-3 ${currentResult.sentiment.toLowerCase()}`}>
                    {currentResult.sentiment}
                  </div>
                  <div className="confidence-bar w-100 mt-4">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${currentResult.confidence * 100}%`,
                        backgroundColor: getConfidenceColor(currentResult.sentiment)
                      }}
                    ></div>
                  </div>
                  <p className="mt-2 text-secondary">
                    Confidence: {(currentResult.confidence * 100).toFixed(1)}%
                  </p>

                  {currentResult.explanation && (
                    <div className="explanation-box mt-4 p-3 rounded-4 w-100" style={{ background: 'rgba(0,0,0,0.2)', textAlign: 'left' }}>
                      <p className="small text-secondary mb-2 fw-bold">Word Importance:</p>
                      <div className="d-flex flex-wrap gap-1">
                        {currentResult.explanation.map((item, idx) => (
                          <span
                            key={idx}
                            className="px-1 rounded"
                            style={{
                              backgroundColor: getConfidenceColor(currentResult.sentiment) + (Math.round(item.score * 80).toString(16).padStart(2, '0')),
                              borderBottom: `2px solid ${getConfidenceColor(currentResult.sentiment)}${Math.round(item.score * 255).toString(16).padStart(2, '0')}`,
                              color: 'white',
                              fontSize: '0.9rem',
                              wordBreak: 'break-word',
                              maxWidth: '100%'
                            }}
                            title={`Score: ${item.score}`}
                          >
                            {item.word.replace(/_/g, '')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {currentResult.aspects && currentResult.aspects.length > 0 && (
                    <div className="aspects-box mt-4 p-3 rounded-4 w-100" style={{ background: 'rgba(0,0,0,0.1)', border: '1px solid var(--glass-border)' }}>
                      <p className="small text-secondary mb-3 fw-bold text-start">Aspect Analysis:</p>
                      <div className="d-flex flex-column gap-2">
                        {currentResult.aspects.map((asp, idx) => (
                          <div key={idx} className="d-flex justify-content-between align-items-center bg-black bg-opacity-25 p-2 rounded-3">
                            <span className="fw-medium small text-start">{asp.aspect}</span>
                            <span className={`sentiment-badge py-1 px-2 ${asp.sentiment.toLowerCase()}`} style={{ fontSize: '0.65rem' }}>
                              {asp.sentiment}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : history.length > 0 ? (
                <div className="w-100">
                  <h3 className="h5 mb-4 text-start d-flex align-items-center text-info">
                    <History className="me-2" size={20} /> Latest Activities
                  </h3>
                  <div className="d-flex flex-column gap-3">
                    {history.slice(0, 5).map((item, idx) => (
                      <div key={idx} className="glass-card p-3 text-start border-start border-4" style={{ borderColor: getConfidenceColor(item.sentiment) }}>
                        <p className="mb-2 small">"{item.text}"</p>
                        <div className="d-flex justify-content-between align-items-center">
                          <span className={`sentiment-badge py-1 px-2 ${item.sentiment.toLowerCase()}`} style={{ fontSize: '0.7rem' }}>
                            {item.sentiment}
                          </span>
                          <span className="small opacity-50">{(item.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-secondary opacity-50">
                  <Sparkles size={64} className="mb-3" />
                  <p>Prediction results will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}

export default App;
