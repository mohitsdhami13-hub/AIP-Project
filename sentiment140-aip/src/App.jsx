import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Brain, Zap, BarChart3, MessageSquare, Sparkles, Activity,
  Server, RotateCcw, TrendingUp, TrendingDown, Target, Award,
  Database, Globe, Layers, BookOpen, Dices, CheckCircle2,
  AlertCircle, ChevronDown, Check, ArrowRight, Menu, Sliders,
  Search, Filter, Info, ShieldCheck, CheckSquare, SlidersHorizontal
} from 'lucide-react';
import {
  analyzeTweet, analyzeBatch, MODEL_METRICS,
  apiClient
} from './sentimentEngine';

// ─── Pre-Built Scenarios matching Inspiration Image ───
const PRESET_TWEETS = [
  {
    id: "praise",
    emoji: "😊",
    label: "Praise",
    tag: "Positive",
    tagClass: "bg-[#C4E7CD] text-[#1E562F]",
    cardBg: "bg-[#D9EEDC]",
    cardBorder: "border-[#B2D8BA]",
    cardHover: "hover:border-[#8CC698]",
    quote: "I absolutely love this new update! It's super fast, intuitive, and works flawlessly.",
    text: "I absolutely love this new update! It's super fast, intuitive, and works flawlessly."
  },
  {
    id: "complaint",
    emoji: "😡",
    label: "Service Issue",
    tag: "Negative",
    tagClass: "bg-[#F8BFB0] text-[#782819]",
    cardBg: "bg-[#FBD5C8]",
    cardBorder: "border-[#F5B5A0]",
    cardHover: "hover:border-[#E88C71]",
    quote: "This is the worst customer support experience I have ever had. Completely unhelpful and slow.",
    text: "This is the worst customer support experience I have ever had. Completely unhelpful and slow."
  },
  {
    id: "negation",
    emoji: "🤔",
    label: "'Not Bad'",
    tag: "Negation",
    tagClass: "bg-[#D6C7F5] text-[#422C79]",
    cardBg: "bg-[#E5DCF8]",
    cardBorder: "border-[#C9B9F0]",
    cardHover: "hover:border-[#A68DE2]",
    quote: "The new laptop is not bad at all, actually quite impressive performance for the price point.",
    text: "The new laptop is not bad at all, actually quite impressive performance for the price point."
  },
  {
    id: "mixed",
    emoji: "😐",
    label: "Mixed Opinion",
    tag: "Mixed",
    tagClass: "bg-[#FCE19B] text-[#71430F]",
    cardBg: "bg-[#FEEEC7]",
    cardBorder: "border-[#F5D483]",
    cardHover: "hover:border-[#EBBF4F]",
    quote: "The camera quality is breathtaking, but the battery life is terrible and drains in three hours.",
    text: "The camera quality is breathtaking, but the battery life is terrible and drains in three hours."
  },
  {
    id: "launch",
    emoji: "🚀",
    label: "AI Feedback",
    tag: "Excited",
    tagClass: "bg-[#BFE0FB] text-[#1E4B75]",
    cardBg: "bg-[#D6E8FA]",
    cardBorder: "border-[#A7CDF3]",
    cardHover: "hover:border-[#7AB1E8]",
    quote: "Just tested the new open source model and wow, this completely changes our workflow for good!",
    text: "Just tested the new open source model and wow, this completely changes our workflow for good!"
  },
  {
    id: "bug",
    emoji: "🚨",
    label: "Crash Report",
    tag: "Bug",
    tagClass: "bg-[#F9BCB5] text-[#7E241B]",
    cardBg: "bg-[#FCD0CB]",
    cardBorder: "border-[#F4A79E]",
    cardHover: "hover:border-[#E47B70]",
    quote: "The app keeps freezing and crashing on launch after the latest patch. So disappointed.",
    text: "The app keeps freezing and crashing on launch after the latest patch. So disappointed."
  }
];

const BATCH_PRESETS = {
  '#AI_Product_Launch': [
    "Just got the new AI assistant and it's absolutely incredible! Best purchase ever!",
    "The AI features are really disappointing. Nothing works as advertised.",
    "Not bad for a first release, looking forward to updates and improvements!",
    "This AI product is a total game changer for productivity! Love it!",
    "Horrible user interface. The AI barely understands basic commands. Waste of money.",
    "Pretty impressed with the speed and accuracy. Great job by the dev team!",
    "Can't believe I paid so much for something so broken and buggy.",
    "The voice recognition is amazingly good. Never seen anything like it!",
    "Returning this immediately. Customer service was rude and unhelpful.",
    "Finally an AI tool that actually delivers on its promises! Highly recommend!",
    "Overpriced and overhyped. Saving my money for a better alternative.",
    "The new update fixed everything! Now it works perfectly. Thank you team!"
  ],
  '#AirlineCustomerService': [
    "Thank you for the amazing upgrade! Best flight experience ever!",
    "3 hours delayed with no explanation. Worst airline experience of my life.",
    "The cabin crew was absolutely wonderful and so helpful during the flight.",
    "Lost my luggage AGAIN. This is the third time this year. So frustrated!",
    "Smooth boarding, comfortable seats, and great food. Will fly again!",
    "Never flying with this airline again. Rude staff and terrible service.",
    "Got rebooked on an earlier flight for free. Excellent customer service!",
    "The plane was dirty and the entertainment system was broken. Disgusting.",
    "First class lounge was beautiful. Felt so pampered and relaxed!",
    "Cancelled my flight with zero notice. Absolutely unacceptable!"
  ],
  '#CinemaNightReviews': [
    "Just watched the new movie and it was absolutely fantastic! Best film this year!",
    "What a boring and predictable storyline. Total waste of time and money.",
    "The special effects were incredible! A visual masterpiece worth every penny.",
    "Terrible acting and even worse dialogue. So disappointed in this sequel.",
    "Loved every minute of it! The plot twist at the end was brilliant!",
    "The movie was way too long and dragged on forever. Almost fell asleep.",
    "Amazing soundtrack and cinematography! A true work of art.",
    "Walked out halfway through. Never been so bored at a cinema.",
    "This director never disappoints! Another stunning masterpiece.",
    "Overrated and overhyped. Don't believe the reviews. Save your money."
  ]
};

// ─── Stat Card 1: 3/4 Circular Radial Progress SVG ───
function AccuracyBadge() {
  const r = 24;
  const c = 2 * Math.PI * r;
  const progress = 0.7984;
  return (
    <div className="relative w-14 h-14 flex items-center justify-center shrink-0">
      <svg width="56" height="56" viewBox="0 0 56 56" className="-rotate-90">
        <circle cx="28" cy="28" r={r} fill="none" stroke="#E3ECE6" strokeWidth="5.5" />
        <circle
          cx="28" cy="28" r={r} fill="none" stroke="#96B8A2" strokeWidth="5.5"
          strokeDasharray={`${c * progress} ${c}`}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-[12px] font-black text-[#2D231E]">3/4</span>
      </div>
    </div>
  );
}

// ─── Stat Card 2: Smooth Area Wave Chart SVG ───
function RocAucWave() {
  return (
    <div className="w-18 h-12 flex items-center justify-center shrink-0">
      <svg width="72" height="40" viewBox="0 0 72 40" fill="none">
        <defs>
          <linearGradient id="peachGrad" x1="0" y1="0" x2="0" y2="40" gradientUnits="userSpaceOnUse">
            <stop stopColor="#F9C8BA" stopOpacity="0.85" />
            <stop offset="1" stopColor="#FDF2EE" stopOpacity="0.15" />
          </linearGradient>
        </defs>
        <path
          d="M2 36 C 14 34, 18 8, 28 8 C 38 8, 42 26, 52 26 C 62 26, 65 4, 70 4 L 70 38 L 2 38 Z"
          fill="url(#peachGrad)"
        />
        <path
          d="M2 36 C 14 34, 18 8, 28 8 C 38 8, 42 26, 52 26 C 62 26, 65 4, 70 4"
          stroke="#E58A73"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

// ─── Stat Card 3: Speedometer Gauge SVG ───
function SpeedometerGauge() {
  return (
    <div className="w-18 h-12 flex items-center justify-center shrink-0">
      <svg width="72" height="42" viewBox="0 0 72 42" fill="none">
        <defs>
          <linearGradient id="speedGrad" x1="0" y1="0" x2="72" y2="0" gradientUnits="userSpaceOnUse">
            <stop stopColor="#E48873" />
            <stop offset="0.5" stopColor="#F5D483" />
            <stop offset="1" stopColor="#8DB99A" />
          </linearGradient>
        </defs>
        <path
          d="M 8 38 A 28 28 0 0 1 64 38"
          fill="none"
          stroke="url(#speedGrad)"
          strokeWidth="6"
          strokeLinecap="round"
        />
        {/* Fast needle pointed to <2.5ms green zone */}
        <line x1="36" y1="38" x2="55" y2="19" stroke="#2D231E" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="36" cy="38" r="3.5" fill="#2D231E" />
      </svg>
    </div>
  );
}

// ─── Stat Card 4: Sticky Notes SVG ───
function StickyNotes() {
  return (
    <div className="w-16 h-12 flex items-center justify-center shrink-0">
      <svg width="56" height="44" viewBox="0 0 56 44" fill="none">
        {/* Background Note (Lilac) */}
        <g transform="rotate(-7, 24, 22)">
          <rect x="10" y="6" width="28" height="30" rx="3" fill="#DDD0F8" stroke="#C3B3E8" strokeWidth="1.5" />
          <text x="14" y="22" fill="#583C8E" fontSize="9" fontWeight="800" fontFamily="sans-serif">w</text>
        </g>
        {/* Foreground Note (Mint Sage) */}
        <g transform="rotate(6, 32, 24)">
          <rect x="22" y="8" width="28" height="30" rx="3" fill="#D3EBDC" stroke="#B0DAC0" strokeWidth="1.5" />
          <text x="26" y="24" fill="#235A36" fontSize="8" fontWeight="800" fontFamily="sans-serif">word</text>
        </g>
      </svg>
    </div>
  );
}

// ─── Cute Gauge for Classification Outcome ───
function SentimentArcGauge({ probabilityPositive }) {
  const prob = typeof probabilityPositive === 'number' ? probabilityPositive : 0.745;
  const angle = -90 + prob * 180;

  return (
    <div className="flex flex-col items-center justify-center shrink-0">
      <svg width="150" height="85" viewBox="0 0 140 80" className="overflow-visible">
        <defs>
          <linearGradient id="gaugePeachGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#E2A79A" />
            <stop offset="100%" stopColor="#ECC2B8" />
          </linearGradient>
          <linearGradient id="gaugeGreenGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#BBDBC6" />
            <stop offset="100%" stopColor="#96B8A2" />
          </linearGradient>
        </defs>

        {/* Negative Arc (Left half) */}
        <path
          d="M 18 68 A 52 52 0 0 1 70 16"
          fill="none"
          stroke="url(#gaugePeachGrad)"
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* Positive Arc (Right half) */}
        <path
          d="M 70 16 A 52 52 0 0 1 122 68"
          fill="none"
          stroke="url(#gaugeGreenGrad)"
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* Center Needle */}
        <g transform={`rotate(${angle}, 70, 68)`} style={{ transition: 'transform 0.5s cubic-bezier(0.16, 1, 0.3, 1)' }}>
          <line x1="70" y1="68" x2="70" y2="22" stroke="#2D231E" strokeWidth="2.5" strokeLinecap="round" />
          <polygon points="67.5,28 70,18 72.5,28" fill="#2D231E" />
        </g>

        {/* Pivot Hub */}
        <circle cx="70" cy="68" r="4.5" fill="#2D231E" />
        <circle cx="70" cy="68" r="1.5" fill="#FFFDFB" />
      </svg>

      <div className="text-center mt-1">
        <span className="text-xl font-black text-[#2D231E] tracking-tight tabular-nums">
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

// ─── Cute Decision Mechanics Line Chart & Toggle ───
function DecisionMechanicsWidget() {
  const [toggleActive, setToggleActive] = useState(true);

  return (
    <div className="space-y-2 pt-1">
      <div className="flex items-center justify-between gap-4">
        {/* Dual Wave Curves SVG */}
        <div className="w-28 h-9 flex items-center justify-center shrink-0">
          <svg width="105" height="34" viewBox="0 0 105 34" fill="none">
            <path
              d="M 2 28 C 16 28, 24 6, 38 6 C 52 6, 60 26, 74 26 C 88 26, 94 4, 103 4"
              stroke="#E48873"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
            <path
              d="M 2 16 C 16 16, 24 30, 38 30 C 52 30, 60 10, 74 10 C 88 10, 94 22, 103 22"
              stroke="#8DB99A"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          </svg>
        </div>

        {/* Separator Line with Decision Dots */}
        <div className="flex-1 flex items-center justify-center gap-2">
          <div className="h-0.5 w-6 bg-[#DDD0C3]" />
          <div className="w-2.5 h-2.5 rounded-full bg-[#E78C79] border border-[#FFFDFB] shadow-xs" title="Class 0: Negative" />
          <div className="w-2.5 h-2.5 rounded-full bg-[#96B8A2] border border-[#FFFDFB] shadow-xs" title="Class 1: Positive" />
          <div className="h-0.5 w-6 bg-[#DDD0C3]" />
        </div>

        {/* Toggle Switch */}
        <button
          onClick={() => setToggleActive(!toggleActive)}
          className={`w-11 h-6 rounded-full transition-colors p-0.5 cursor-pointer flex items-center shrink-0 ${
            toggleActive ? 'bg-[#7E6D62]' : 'bg-[#DDD0C3]'
          }`}
          title="Toggle Sigmoid Normalization"
        >
          <div
            className={`w-5 h-5 rounded-full bg-[#FFFDFB] shadow-sm transition-transform duration-200 ${
              toggleActive ? 'translate-x-5' : 'translate-x-0'
            }`}
          />
        </button>
      </div>

      {/* Metadata / Formulas row */}
      <div className="flex items-center justify-between text-xs text-[#7E6D62] font-medium pt-1 px-1">
        <span>Sigmoid: <span className="font-semibold text-[#2D231E]">{toggleActive ? 'σ(z) active' : 'raw logits'}</span></span>
        <span>L2 Reg: <span className="font-semibold text-[#2D231E]">λ = 1.0</span></span>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0); // 0: Sandbox, 1: Batch, 2: Metrics
  const [tweet, setTweet] = useState(PRESET_TWEETS[0].text);
  const [selectedPresetId, setSelectedPresetId] = useState(PRESET_TWEETS[0].id);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Batch analysis state
  const [selectedBatchKey, setSelectedBatchKey] = useState('#AI_Product_Launch');
  const [batchResults, setBatchResults] = useState([]);
  const [batchSummary, setBatchSummary] = useState(null);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchFilter, setBatchFilter] = useState('all');
  const [batchSearch, setBatchSearch] = useState('');

  // Model Metrics interactive threshold slider
  const [simThreshold, setSimThreshold] = useState(0.50);

  // Backend state
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const [backendConnected, setBackendConnected] = useState(false);
  const [showApiModal, setShowApiModal] = useState(false);

  const textareaRef = useRef(null);
  const metrics = MODEL_METRICS || {};

  useEffect(() => {
    checkBackend(backendUrl);
  }, []);

  useEffect(() => {
    handleAnalyze(PRESET_TWEETS[0].text);
  }, []);

  const checkBackend = useCallback(async (url) => {
    try {
      const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
      if (resp.ok) {
        setBackendConnected(true);
        apiClient.baseUrl = url;
        return;
      }
    } catch {
      // Offline fallback
    }
    setBackendConnected(false);
  }, []);

  const handleAnalyze = async (overrideText) => {
    const textToAnalyze = (overrideText !== undefined ? overrideText : tweet).trim();
    if (!textToAnalyze) return;

    setIsAnalyzing(true);
    try {
      await new Promise(r => setTimeout(r, 60));
      const res = await analyzeTweet(textToAnalyze);
      setResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handlePresetSelect = (preset) => {
    setSelectedPresetId(preset.id);
    setTweet(preset.text);
    handleAnalyze(preset.text);
  };

  const handleRandomTweet = () => {
    const otherPresets = PRESET_TWEETS.filter(p => p.id !== selectedPresetId);
    const chosen = otherPresets[Math.floor(Math.random() * otherPresets.length)] || PRESET_TWEETS[0];
    handlePresetSelect(chosen);
  };

  const handleClear = () => {
    setTweet('');
    setSelectedPresetId(null);
    setResult(null);
    if (textareaRef.current) textareaRef.current.focus();
  };

  const handleRunBatch = async (batchKey) => {
    const tweets = BATCH_PRESETS[batchKey];
    if (!tweets) return;

    setIsBatchRunning(true);
    try {
      const { results, summary } = await analyzeBatch(tweets);
      setBatchResults(results);
      setBatchSummary(summary);
    } finally {
      setIsBatchRunning(false);
    }
  };

  useEffect(() => {
    if (activeTab === 1 && batchResults.length === 0) {
      handleRunBatch(selectedBatchKey);
    }
  }, [activeTab]);

  const filteredBatchResults = useMemo(() => {
    return batchResults.filter((item) => {
      if (batchFilter === 'positive' && item.sentiment !== 'Positive') return false;
      if (batchFilter === 'negative' && item.sentiment !== 'Negative') return false;
      const text = item.tweet || item.text || '';
      if (batchSearch && !text.toLowerCase().includes(batchSearch.toLowerCase())) return false;
      return true;
    });
  }, [batchResults, batchFilter, batchSearch]);

  const displayedTokens = useMemo(() => {
    return result?.token_contributions || [];
  }, [result]);

  return (
    <div className="w-full min-h-screen bg-[#F6F0EA] text-[#2D231E] flex flex-col justify-between selection:bg-[#D6C7F5] selection:text-[#2D231E]">
      {/* ─── Top Navigation Bar matching specification ─── */}
      <header className="w-full border-b border-[#E5E0D8] bg-[#F6F0EA]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-8 h-16 flex items-center justify-between gap-6">
          {/* Brand & Identity */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-[#EADFD3] border border-[#D5C7B8] flex items-center justify-center text-lg shrink-0 shadow-2xs">
              🧠
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-extrabold text-[#2D231E] tracking-tight leading-none">Sentiment140</span>
                <span className="inline-flex items-center text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-[#EDE1D6] text-[#6E5D52] border border-[#DAC9BA] leading-none">
                  AIP-11
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-[#7E6D62] font-medium leading-none mt-1">
                <span>Tweet Sentiment Intelligence</span>
                <span className="text-[#C5B5A5]">•</span>
                <span>Mohit, Vyom, Brian</span>
              </div>
            </div>
          </div>

          {/* Center Pill Navigation Tabs */}
          <nav className="flex items-center gap-2 p-1.5 bg-[#EDE8E0] rounded-xl border border-[#DDD6CC] shadow-2xs">
            {[
              { id: 0, label: 'Live Sandbox', icon: MessageSquare },
              { id: 1, label: 'Batch Simulator', icon: Layers },
              { id: 2, label: 'Model Metrics', icon: Activity }
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-all cursor-pointer leading-none ${
                    isActive
                      ? 'bg-[#FFFDFB] text-[#2D231E] font-bold shadow-xs border border-[#D5C7B8]'
                      : 'text-[#6E5D52] hover:text-[#2D231E] hover:bg-[#F6EFE8]'
                  }`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-[#8DA4EF]' : 'text-[#7E6D62]'}`} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Status Badge & Settings */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setShowApiModal(!showApiModal)}
              className="px-3.5 py-2 rounded-xl bg-[#FFFDFB] border border-[#D5C7B8] hover:border-[#8E7E73] transition-colors cursor-pointer text-xs font-bold text-[#2D231E] shadow-2xs inline-flex items-center gap-2 leading-none"
              title="Configure FastAPI endpoint"
            >
              <span className={`w-2 h-2 rounded-full shrink-0 ${backendConnected ? 'bg-emerald-500 animate-pulse' : 'bg-emerald-600'}`} />
              <span>{backendConnected ? 'FastAPI Online' : 'Local Engine'}</span>
            </button>

            <button
              onClick={() => setShowApiModal(!showApiModal)}
              className="w-9 h-9 rounded-xl bg-[#FFFDFB] border border-[#D5C7B8] inline-flex items-center justify-center text-[#2D231E] hover:border-[#8E7E73] cursor-pointer shadow-2xs"
              title="Model & Engine Settings"
            >
              <Menu className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* ─── API Connection Drawer ─── */}
      {showApiModal && (
        <div className="max-w-[1400px] mx-auto w-full px-8 pt-4">
          <div className="retro-card p-4 space-y-2 bg-[#FFFDFB] border-2 border-[#8DA4EF]/50">
            <div className="flex items-center justify-between">
              <span className="text-xs font-black uppercase text-[#2D231E] tracking-wider flex items-center gap-1.5">
                <Server className="w-4 h-4 text-[#8DA4EF]" />
                FastAPI Backend Connection
              </span>
              <button
                onClick={() => setShowApiModal(false)}
                className="text-xs font-bold text-[#7E6D62] hover:text-[#2D231E]"
              >
                ✕ Close
              </button>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 text-xs border border-[#D5C7B8] rounded-xl px-3 py-2 font-mono text-[#2D231E] bg-[#FBF8F4] focus:outline-none focus:border-[#8DA4EF]"
              />
              <button
                onClick={() => checkBackend(backendUrl)}
                className="btn-editorial"
              >
                Test Ping
              </button>
            </div>
            <p className="text-xs text-[#7E6D62]">
              Status: {backendConnected ? 'Connected to local FastAPI backend.' : 'Using client-side vectorizer (50k vocabulary & 250+ sentiment features).'}
            </p>
          </div>
        </div>
      )}

      {/* ─── Main Page Centered Container matching specification ─── */}
      <main className="max-w-[1400px] mx-auto px-8 pt-6 pb-20 w-full flex-1 flex flex-col gap-6">
        {/* Top Stat Row with 4 Illustrated Cards (Shared Across Tabs) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Card 1: TEST ACCURACY */}
          <div className="retro-card p-6 flex items-center justify-between gap-4">
            <div className="flex flex-col justify-center">
              <div className="text-xs font-bold uppercase tracking-wider text-[#7E6D62] mb-1">TEST ACCURACY</div>
              <div className="text-3xl font-black text-[#2D231E] tracking-tight leading-none my-1 tabular-nums">79.84%</div>
              <div className="text-xs text-[#7E6D62] font-medium mt-1">20,000 test set</div>
            </div>
            <AccuracyBadge />
          </div>

          {/* Card 2: ROC-AUC SCORE */}
          <div className="retro-card p-6 flex items-center justify-between gap-4">
            <div className="flex flex-col justify-center">
              <div className="text-xs font-bold uppercase tracking-wider text-[#7E6D62] mb-1">ROC-AUC SCORE</div>
              <div className="text-3xl font-black text-[#2D231E] tracking-tight leading-none my-1 tabular-nums">87.52%</div>
              <div className="text-xs text-[#7E6D62] font-medium mt-1">High discrimination</div>
            </div>
            <RocAucWave />
          </div>

          {/* Card 3: INFERENCE LATENCY */}
          <div className="retro-card p-6 flex items-center justify-between gap-4">
            <div className="flex flex-col justify-center">
              <div className="text-xs font-bold uppercase tracking-wider text-[#7E6D62] mb-1">INFERENCE LATENCY</div>
              <div className="text-3xl font-black text-[#2D231E] tracking-tight leading-none my-1 tabular-nums">&lt; 2.5 ms</div>
              <div className="text-xs text-[#7E6D62] font-medium mt-1">Instant scoring</div>
            </div>
            <SpeedometerGauge />
          </div>

          {/* Card 4: MODEL VOCABULARY */}
          <div className="retro-card p-6 flex items-center justify-between gap-4">
            <div className="flex flex-col justify-center">
              <div className="text-xs font-bold uppercase tracking-wider text-[#7E6D62] mb-1">MODEL VOCABULARY</div>
              <div className="text-3xl font-black text-[#2D231E] tracking-tight leading-none my-1 tabular-nums">50,000</div>
              <div className="text-xs text-[#7E6D62] font-medium mt-1">Sublinear TF-IDF</div>
            </div>
            <StickyNotes />
          </div>
        </div>

        {/* ═══ TAB 0: LIVE SANDBOX ═══ */}
        {activeTab === 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch flex-1">
            {/* Left Column (lg:col-span-7 flex flex-col gap-6) */}
            <div className="lg:col-span-7 flex flex-col gap-6">
              {/* Tweet Sentiment Studio Card */}
              <div className="retro-card p-6 flex flex-col justify-between gap-4 flex-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-[#7E6D62]" />
                    <h2 className="text-sm font-bold text-[#2D231E] tracking-tight leading-none">Tweet Sentiment Studio</h2>
                  </div>
                  <span className="text-xs font-mono font-bold text-[#7E6D62]">
                    {tweet.length} / 280
                  </span>
                </div>

                <textarea
                  ref={textareaRef}
                  value={tweet}
                  onChange={(e) => {
                    setTweet(e.target.value.slice(0, 280));
                    setSelectedPresetId(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                      handleAnalyze();
                    }
                  }}
                  placeholder="Write or paste a tweet to evaluate its sentiment in real-time..."
                  rows={3}
                  className="w-full text-base text-[#2D231E] placeholder-[#9E8D81] bg-[#FFFDFB] border border-[#DDD0C3] rounded-xl p-4 focus:outline-none focus:border-[#8DA4EF] focus:ring-2 focus:ring-[#8DA4EF]/25 transition-all resize-none leading-relaxed flex-1 min-h-[95px]"
                />

                {/* Buttons Row */}
                <div className="flex items-center justify-between gap-3 pt-1">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleRandomTweet}
                      className="btn-editorial"
                    >
                      <Dices className="w-4 h-4 text-[#5A4B40] shrink-0" />
                      <span>Random Sample</span>
                    </button>

                    <button
                      onClick={handleClear}
                      className="btn-editorial"
                    >
                      <RotateCcw className="w-4 h-4 text-[#5A4B40] shrink-0" />
                      <span>Clear</span>
                    </button>
                  </div>

                  <button
                    onClick={() => handleAnalyze()}
                    disabled={!tweet.trim() || isAnalyzing}
                    className="btn-analyze"
                  >
                    {isAnalyzing ? (
                      <>
                        <Sparkles className="w-4 h-4 animate-spin text-[#172142] shrink-0" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 text-[#172142] shrink-0" />
                        <span>Analyze Tweet</span>
                        <kbd className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-[#FFFDFB]/40 text-[#172142] font-semibold border border-[#172142]/20 inline-flex items-center justify-center leading-none ml-2.5 shadow-2xs">Ctrl+Enter</kbd>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Pre-Built Scenarios & 4-Stage Pipeline Sub-Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                {/* Pre-Built Scenarios Card */}
                <div className="retro-card p-6 flex flex-col justify-between gap-3">
                  <h3 className="text-xs font-bold text-[#7E6D62] uppercase tracking-wider leading-none">
                    Pre-Built Scenarios
                  </h3>

                  <div className="grid grid-cols-2 gap-2.5 flex-1">
                    {PRESET_TWEETS.map((preset) => {
                      const isSelected = selectedPresetId === preset.id;
                      return (
                        <button
                          key={preset.id}
                          onClick={() => handlePresetSelect(preset)}
                          className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between gap-2 min-h-[88px] shadow-2xs hover:shadow-xs ${preset.cardBg} ${preset.cardBorder} ${preset.cardHover} ${
                            isSelected ? 'ring-2 ring-[#7B93E0] shadow-xs' : ''
                          }`}
                        >
                          <div className="flex items-center justify-between w-full gap-1.5">
                            <span className="text-xs font-bold text-[#2D231E] flex items-center gap-1.5 min-w-0 leading-none">
                              <span className="text-sm shrink-0 leading-none">{preset.emoji}</span>
                              <span className="truncate">{preset.label}</span>
                            </span>
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md shrink-0 leading-none uppercase tracking-wider ${preset.tagClass}`}>
                              {preset.tag}
                            </span>
                          </div>
                          <p className="line-clamp-2 text-xs leading-relaxed text-[#55463C] italic">
                            "{preset.quote}"
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* 4-Stage NLP Pipeline Breakdown Card */}
                <div className="retro-card p-6 flex flex-col justify-between gap-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">📋</span>
                      <h3 className="text-xs font-bold text-[#7E6D62] uppercase tracking-wider leading-none">
                        4-Stage NLP Pipeline Breakdown
                      </h3>
                    </div>
                  </div>

                  <div className="space-y-2 flex-1 flex flex-col justify-between">
                    {/* Step 1: Raw Input */}
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 rounded-full bg-[#B4D5C1] text-[#1E562F] font-black text-xs flex items-center justify-center shrink-0 border border-[#98C4A5] leading-none shadow-2xs">
                        1
                      </div>
                      <div className="flex-1 min-w-0 px-3.5 py-2 rounded-xl bg-[#FFFDFB] border border-[#DDD0C3] shadow-2xs">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-[#2D231E] block leading-none mb-1">Raw Input</span>
                        <p className="text-xs text-[#5A4B40] truncate font-medium">
                          {result?.preprocessing_steps?.[0]?.output || tweet}
                        </p>
                      </div>
                    </div>

                    {/* Step 2: Regex Cleaning */}
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 rounded-full bg-[#FCE1A9] text-[#71430F] font-black text-xs flex items-center justify-center shrink-0 border border-[#F5D483] leading-none shadow-2xs">
                        2
                      </div>
                      <div className="flex-1 min-w-0 px-3.5 py-2 rounded-xl bg-[#FFF9E6] border border-[#F5E2A8] shadow-2xs">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-[#71430F] block leading-none mb-1">Regex Cleaning</span>
                        <p className="text-xs text-[#5A4B40] truncate font-medium">
                          {result?.preprocessing_steps?.[1]?.output || 'cleaned lowercase without urls mentions'}
                        </p>
                      </div>
                    </div>

                    {/* Step 3: Negation-Safe Stopword Filter */}
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 rounded-full bg-[#D8C9F3] text-[#422C79] font-black text-xs flex items-center justify-center shrink-0 border border-[#C5B3E8] leading-none shadow-2xs">
                        3
                      </div>
                      <div className="flex-1 min-w-0 px-3.5 py-2 rounded-xl bg-[#F6F1FD] border border-[#E0D1F8] shadow-2xs">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-[#422C79] block leading-none mb-1">Negation-Safe Filter</span>
                        <p className="text-xs text-[#5A4B40] truncate font-medium">
                          {result?.preprocessing_steps?.[2]?.output || 'stopwords filtered negations preserved'}
                        </p>
                      </div>
                    </div>

                    {/* Step 4: Porter Stemmed N-Grams */}
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 rounded-full bg-[#F9C9C2] text-[#7E241B] font-black text-xs flex items-center justify-center shrink-0 border border-[#F4A79E] leading-none shadow-2xs">
                        4
                      </div>
                      <div className="flex-1 min-w-0 px-3.5 py-2 rounded-xl bg-[#FEF2F0] border border-[#F9D0CB] shadow-2xs">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-[#7E241B] block leading-none mb-1">Porter Stemmed N-Grams</span>
                        <p className="text-xs text-[#5A4B40] truncate font-medium">
                          {result?.preprocessing_steps?.[3]?.output || 'stemmed token ngrams'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column (lg:col-span-5 flex flex-col) */}
            <div className="lg:col-span-5 retro-card p-6 flex flex-col justify-between gap-6 h-full">
              {/* ── Sub-Section 1: Classification Outcome ── */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-[#7E6D62]">
                    Classification Outcome
                  </span>
                  <span className="text-xs font-semibold text-[#7E6D62]">
                    {result?.source === 'api' ? 'FastAPI Backend' : 'Client LR Model'}
                  </span>
                </div>

                <div className="flex items-center justify-between gap-6 pt-1">
                  <SentimentArcGauge probabilityPositive={result?.probability_positive} />
                  <div className="text-right flex-1 min-w-0 flex flex-col items-end justify-center">
                    <div className={`text-xl font-black tracking-tight leading-tight ${result?.sentiment === 'Negative' ? 'text-[#8E281F]' : 'text-[#1E562F]'}`}>
                      {result?.sentiment === 'Negative' ? 'Negative Sentiment' : 'Positive Sentiment'}
                    </div>
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#EFE5DB]/80 border border-[#DDD0C3] mt-2.5">
                      <span className="text-[10px] font-bold text-[#7E6D62] uppercase tracking-wider">Confidence</span>
                      <span className="text-sm font-black text-[#2D231E] tabular-nums">
                        {((result?.confidence || 0.745) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-[#7E6D62] mt-1.5 tabular-nums">
                      P(y=1) = {((result?.probability_positive || 0.745) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Threshold Caption */}
                <div className="text-xs text-[#7E6D62] font-semibold text-center flex items-center justify-center gap-2 pt-1">
                  <span>0.0 (Pure Negative)</span>
                  <span className="text-[#C5B5A5]">•</span>
                  <span>Threshold <span className="font-bold text-[#2D231E]">τ = 0.50</span></span>
                  <span className="text-[#C5B5A5]">•</span>
                  <span>1.0 (Pure Positive)</span>
                </div>

                {/* Dual Probability Split Bar */}
                <div className="flex items-center rounded-lg overflow-hidden h-6 text-xs font-bold shadow-2xs border border-[#DDD0C3]">
                  <div
                    className="bg-[#E2A79A] text-[#4A1D15] h-full flex items-center px-3.5 transition-all duration-500 tabular-nums font-extrabold"
                    style={{ width: `${Math.max(((result?.probability_negative || 0.255) * 100), 22).toFixed(1)}%` }}
                  >
                    Negative {((result?.probability_negative || 0.255) * 100).toFixed(1)}%
                  </div>
                  <div
                    className="bg-[#96B8A2] text-[#13381E] h-full flex items-center justify-end px-3.5 transition-all duration-500 flex-1 tabular-nums font-extrabold"
                  >
                    Positive {((result?.probability_positive || 0.745) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* ── Sub-Section 2: Influential Words (Explainable AI) ── */}
              <div className="border-t border-[#E5E0D8] pt-5 space-y-3 flex-1 flex flex-col justify-center">
                <div className="flex items-center justify-between pb-1">
                  <span className="text-xs font-bold text-[#2D231E] uppercase tracking-wider">
                    Influential Words (Explainable AI)
                  </span>
                  <span className="text-xs font-mono text-[#7E6D62] font-medium">
                    Impact = TF-IDF · θ
                  </span>
                </div>

                <div className="space-y-2">
                  {displayedTokens.length > 0 ? (
                    displayedTokens.slice(0, 6).map((tok, i) => {
                      const isPos = tok.impact_score > 0;
                      const barWidth = Math.min(Math.abs(tok.impact_score) * 85 + 20, 100);
                      return (
                        <div key={i} className="flex items-center justify-between gap-3 text-xs h-6">
                          <span className="w-[90px] font-mono text-xs font-bold text-[#2D231E] truncate">
                            {tok.token}
                          </span>
                          <span className={`w-[60px] font-mono text-xs font-bold text-right tabular-nums ${isPos ? 'text-[#2E7D5B]' : 'text-[#C84B31]'}`}>
                            {tok.impact_score > 0 ? `+${tok.impact_score.toFixed(3)}` : tok.impact_score.toFixed(3)}
                          </span>
                          <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden self-center">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isPos ? 'bg-[#2E7D5B]' : 'bg-[#C84B31]'
                              }`}
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                          <span className="w-[65px] text-right font-mono text-xs text-[#7E6D62] tabular-nums">
                            wt: {tok.tfidf_weight}
                          </span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-xs text-[#7E6D62] text-center py-2 italic">
                      Analyzing influential vocabulary...
                    </div>
                  )}
                </div>
              </div>

              {/* ── Sub-Section 3: Logistic Regression Decision Mechanics ── */}
              <div className="border-t border-[#E5E0D8] pt-5 space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-[#2D231E] uppercase tracking-wider">
                    Logistic Regression Decision Mechanics
                  </h3>
                  <span className="text-xs font-mono font-medium text-[#7E6D62]">
                    z = w · x + b
                  </span>
                </div>
                <DecisionMechanicsWidget />
              </div>
            </div>
          </div>
        )}

        {/* ═══ TAB 1: BATCH SIMULATOR ═══ */}
        {activeTab === 1 && (
          <div className="flex flex-col gap-6 w-full flex-1">
            {/* Header & Controls Card */}
            <div className="retro-card p-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-black text-[#2D231E] flex items-center gap-2">
                    <Layers className="w-5 h-5 text-[#8DA4EF]" />
                    Batch Tweet Stream Analysis
                  </h2>
                  <p className="text-xs text-[#7E6D62] mt-1">
                    Process simulated live streams across high-volume Twitter topics to measure brand sentiment dynamics.
                  </p>
                </div>

                <button
                  onClick={() => handleRunBatch(selectedBatchKey)}
                  disabled={isBatchRunning}
                  className="btn-analyze self-start sm:self-auto"
                >
                  {isBatchRunning ? <Sparkles className="w-4 h-4 animate-spin text-[#172142] shrink-0" /> : <Zap className="w-4 h-4 text-[#172142] shrink-0" />}
                  <span>Run Batch Stream</span>
                </button>
              </div>

              {/* Topic Selectors */}
              <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-[#E5E0D8]">
                <span className="text-xs font-bold text-[#7E6D62]">Select Topic Stream:</span>
                {Object.keys(BATCH_PRESETS).map((key) => {
                  const isSel = selectedBatchKey === key;
                  return (
                    <button
                      key={key}
                      onClick={() => {
                        setSelectedBatchKey(key);
                        handleRunBatch(key);
                      }}
                      className={`px-4 py-2 rounded-full text-sm font-medium border transition-all cursor-pointer ${
                        isSel
                          ? 'bg-[#E5DCF8] border-[#C9B9F0] text-[#422C79] font-bold shadow-xs'
                          : 'bg-[#FFFDFB] border-[#DDD0C3] text-[#6E5D52] hover:border-[#8E7E73]'
                      }`}
                    >
                      {key}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Batch KPI Summary */}
            {batchSummary && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="retro-card p-6 flex flex-col justify-center">
                  <span className="text-xs font-bold text-[#7E6D62] uppercase tracking-wider">Stream Sample</span>
                  <p className="text-3xl font-black text-[#2D231E] leading-none mt-1.5 tabular-nums">{batchSummary.total}</p>
                  <p className="text-xs text-[#7E6D62] font-semibold mt-1">Tweets evaluated</p>
                </div>

                <div className="retro-card p-6 flex flex-col justify-center">
                  <span className="text-xs font-bold text-[#7E6D62] uppercase tracking-wider">Avg Confidence</span>
                  <p className="text-3xl font-black text-[#7B93E0] leading-none mt-1.5 tabular-nums">
                    {(((batchSummary.average_confidence ?? batchSummary.avgConfidence ?? 0.72)) * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-[#7E6D62] font-semibold mt-1">Model certainty</p>
                </div>

                <div className="retro-card p-6 flex flex-col justify-center">
                  <span className="text-xs font-bold text-[#7E6D62] uppercase tracking-wider">Positive Share</span>
                  <p className="text-3xl font-black text-[#1E562F] leading-none mt-1.5 tabular-nums">
                    {((batchSummary.positive / batchSummary.total) * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-[#7E6D62] font-semibold mt-1">{batchSummary.positive} positive tweets</p>
                </div>

                <div className="retro-card p-6 flex flex-col justify-center">
                  <span className="text-xs font-bold text-[#7E6D62] uppercase tracking-wider">Negative Share</span>
                  <p className="text-3xl font-black text-[#782819] leading-none mt-1.5 tabular-nums">
                    {((batchSummary.negative / batchSummary.total) * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-[#7E6D62] font-semibold mt-1">{batchSummary.negative} negative tweets</p>
                </div>
              </div>
            )}

            {/* Overall Sentiment Distribution Visual Bar */}
            {batchSummary && (
              <div className="retro-card p-6 space-y-2">
                <div className="flex items-center justify-between text-xs font-bold text-[#2D231E]">
                  <span>Sentiment Distribution Balance</span>
                  <span className="font-semibold text-[#7E6D62] tabular-nums">{batchSummary.total} Total Tweets</span>
                </div>
                <div className="flex items-center rounded-lg overflow-hidden h-6 text-xs font-bold shadow-2xs border border-[#DDD0C3]">
                  <div
                    className="bg-[#96B8A2] text-[#13381E] font-extrabold h-full flex items-center px-4 transition-all duration-500 tabular-nums leading-none"
                    style={{ width: `${((batchSummary.positive / batchSummary.total) * 100).toFixed(1)}%` }}
                  >
                    Positive {((batchSummary.positive / batchSummary.total) * 100).toFixed(1)}% ({batchSummary.positive})
                  </div>
                  <div
                    className="bg-[#E78C79] text-[#4A1D15] font-extrabold h-full flex items-center justify-end px-4 transition-all duration-500 flex-1 tabular-nums leading-none"
                  >
                    Negative {((batchSummary.negative / batchSummary.total) * 100).toFixed(1)}% ({batchSummary.negative})
                  </div>
                </div>
              </div>
            )}

            {/* Stream Records Workbench with Filter & Search */}
            <div className="retro-card p-6 space-y-4 flex flex-col flex-1">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <h3 className="text-xs font-bold text-[#2D231E] uppercase tracking-wider">
                  Stream Records ({filteredBatchResults.length} of {batchResults.length})
                </h3>

                {/* Filter & Search Bar */}
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex items-center gap-1.5 bg-[#EAE4DC] p-1.5 rounded-xl border border-[#DDD6CC]">
                    <button
                      onClick={() => setBatchFilter('all')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                        batchFilter === 'all'
                          ? 'bg-white text-[#2D231E] shadow-xs border border-[#D5C7B8]'
                          : 'text-[#6E5D52] hover:text-[#2D231E] hover:bg-white/50'
                      }`}
                    >
                      All ({batchResults.length})
                    </button>
                    <button
                      onClick={() => setBatchFilter('positive')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                        batchFilter === 'positive'
                          ? 'bg-[#D9EEDC] text-[#1E562F] shadow-xs border border-[#B2D8BA]'
                          : 'text-[#6E5D52] hover:text-[#1E562F] hover:bg-white/50'
                      }`}
                    >
                      Positive ({batchResults.filter(r => r.sentiment === 'Positive').length})
                    </button>
                    <button
                      onClick={() => setBatchFilter('negative')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                        batchFilter === 'negative'
                          ? 'bg-[#FBD5C8] text-[#782819] shadow-xs border border-[#F5B5A0]'
                          : 'text-[#6E5D52] hover:text-[#782819] hover:bg-white/50'
                      }`}
                    >
                      Negative ({batchResults.filter(r => r.sentiment === 'Negative').length})
                    </button>
                  </div>

                  <div className="relative w-64">
                    <Search className="w-4 h-4 text-[#8E7E73] absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      type="text"
                      value={batchSearch}
                      onChange={(e) => setBatchSearch(e.target.value)}
                      placeholder="Search tweets..."
                      style={{ paddingLeft: '2.5rem', paddingRight: '1rem' }}
                      className="w-full py-1.5 text-sm rounded-lg border border-[#DDD6CC] bg-[#FFFDFB] text-[#2D231E] focus:outline-none focus:border-[#7B93E0] focus:ring-1 focus:ring-[#7B93E0] transition-all"
                    />
                  </div>
                </div>
              </div>

              {/* Scrollable Records List */}
              <div className="space-y-2 flex-1 max-h-[480px] overflow-y-auto pr-2">
                {filteredBatchResults.map((item, idx) => {
                  const isPos = item.sentiment === 'Positive';
                  const tweetText = item.tweet || item.text || '';
                  return (
                    <div
                      key={idx}
                      className="px-4 py-3 flex items-center justify-between gap-4 border-b border-[#F0ECE1] hover:bg-[#FAF8F5] transition-colors rounded-lg text-sm"
                    >
                      <div className="flex items-start gap-3 flex-1 min-w-0">
                        <span className="w-6 h-6 rounded-full bg-[#EFE5DB] text-[#7E6D62] text-xs font-bold flex items-center justify-center shrink-0 mt-0.5 tabular-nums">
                          {idx + 1}
                        </span>
                        <p className="text-[#2D231E] leading-relaxed flex-1">
                          "{tweetText}"
                        </p>
                      </div>

                      <div className="flex items-center gap-3 shrink-0">
                        <span
                          className={`w-20 text-center py-1 px-2.5 rounded-full text-xs font-semibold ${
                            isPos
                              ? 'bg-[#D9EEDC] text-[#1E562F] border border-[#B2D8BA]'
                              : 'bg-[#FBD5C8] text-[#782819] border border-[#F5B5A0]'
                          }`}
                        >
                          {item.sentiment}
                        </span>
                        <span className="text-xs font-mono font-bold text-[#7E6D62] w-14 text-right tabular-nums">
                          {((item.confidence || 0.75) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ═══ TAB 2: MODEL METRICS ═══ */}
        {activeTab === 2 && (
          <div className="flex flex-col gap-6 w-full flex-1">
            {/* Top Row: Confusion Matrix (7 cols) + Architecture Specs (5 cols) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              {/* Confusion Matrix (7 cols) */}
              <div className="lg:col-span-7 retro-card p-6 flex flex-col justify-between gap-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-[#2D231E] uppercase tracking-wider flex items-center gap-2">
                    <Target className="w-4 h-4 text-[#8DA4EF]" />
                    Empirical Confusion Matrix (20,000 Holdout Test)
                  </h3>
                  <span className="text-xs font-bold text-[#7E6D62] bg-[#EFE5DB] px-2.5 py-1 rounded-lg">
                    Stanford Sentiment140
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 flex-1">
                  {[
                    { label: 'True Positive (TP)', val: 7912, pct: '39.6%', sub: 'Actual Pos, Pred Pos', bg: 'bg-[#D9EEDC] text-[#1E562F] border-[#B2D8BA]' },
                    { label: 'False Negative (FN)', val: 2088, pct: '10.4%', sub: 'Actual Pos, Pred Neg (Type II)', bg: 'bg-[#FEEEC7] text-[#71430F] border-[#F5D483]' },
                    { label: 'False Positive (FP)', val: 1944, pct: '9.7%', sub: 'Actual Neg, Pred Pos (Type I)', bg: 'bg-[#FBD5C8] text-[#782819] border-[#F5B5A0]' },
                    { label: 'True Negative (TN)', val: 8056, pct: '40.3%', sub: 'Actual Neg, Pred Neg', bg: 'bg-[#E5DCF8] text-[#422C79] border-[#C9B9F0]' },
                  ].map((c, i) => (
                    <div key={i} className={`p-5 rounded-xl border ${c.bg} flex flex-col justify-between min-h-[125px] shadow-2xs`}>
                      <div className="flex justify-between items-center text-xs font-bold">
                        <span>{c.label}</span>
                        <span className="tabular-nums">{c.pct}</span>
                      </div>
                      <p className="text-2xl font-black leading-none my-1 tabular-nums">{c.val.toLocaleString()}</p>
                      <p className="text-xs opacity-80">{c.sub}</p>
                    </div>
                  ))}
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-[#7E6D62] pt-2 border-t border-[#E5E0D8]">
                  <span>Overall Accuracy: <strong className="text-[#1E562F] font-bold tabular-nums">79.84%</strong> (15,968 / 20,000)</span>
                  <span>Balanced Ground Truth: 10k Pos / 10k Neg</span>
                </div>
              </div>

              {/* Model Specifications (5 cols) */}
              <div className="lg:col-span-5 retro-card p-6 flex flex-col justify-between gap-3">
                <div className="flex items-center gap-2 mb-1">
                  <Database className="w-4 h-4 text-[#8DA4EF]" />
                  <h3 className="text-xs font-bold text-[#2D231E] uppercase tracking-wider">
                    Model Architecture & Specs
                  </h3>
                </div>

                <div className="space-y-2 text-xs flex-1 flex flex-col justify-between">
                  <div className="flex justify-between items-center py-1 border-b border-[#E5E0D8]">
                    <span className="text-[#7E6D62]">Corpus Source</span>
                    <span className="font-semibold text-[#2D231E]">Sentiment140 (Stanford)</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-[#E5E0D8]">
                    <span className="text-[#7E6D62]">Working Subset</span>
                    <span className="font-semibold text-[#2D231E]">100,000 tweets (50k / 50k)</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-[#E5E0D8]">
                    <span className="text-[#7E6D62]">Train / Test Split</span>
                    <span className="font-semibold text-[#2D231E]">80,000 / 20,000 (80/20)</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-[#E5E0D8]">
                    <span className="text-[#7E6D62]">Feature Vectorizer</span>
                    <span className="font-semibold text-[#2D231E]">TF-IDF (1-2 N-grams)</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-[#E5E0D8]">
                    <span className="text-[#7E6D62]">Vocabulary Dimension</span>
                    <span className="font-semibold text-[#2D231E]">50,000 features</span>
                  </div>
                  <div className="flex justify-between items-center py-1 border-b border-[#E5E0D8]">
                    <span className="text-[#7E6D62]">Classifier</span>
                    <span className="font-semibold text-[#2D231E]">Logistic Regression (C=1.0)</span>
                  </div>
                  <div className="flex justify-between items-center py-1">
                    <span className="text-[#7E6D62]">Optimization Solver</span>
                    <span className="font-semibold text-[#2D231E]">L-BFGS (L2 reg)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Row: Detailed Classification Report Table & Interactive Threshold */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              {/* Classification Report Table (7 cols) */}
              <div className="lg:col-span-7 retro-card p-6 flex flex-col justify-between gap-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-[#2D231E] uppercase tracking-wider flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-[#8DA4EF]" />
                    Per-Class Performance Breakdown
                  </h3>
                  <span className="text-xs text-[#7E6D62] font-semibold">Macro / Weighted</span>
                </div>

                <div className="overflow-x-auto flex-1">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-[#E5E0D8] text-[11px] font-bold uppercase text-[#7E6D62]">
                        <th className="py-3 px-4">Class</th>
                        <th className="py-3 px-4 text-right">Precision</th>
                        <th className="py-3 px-4 text-right">Recall</th>
                        <th className="py-3 px-4 text-right">F1-Score</th>
                        <th className="py-3 px-4 text-right">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#EFE5DB] text-xs font-semibold tabular-nums">
                      <tr>
                        <td className="py-3 px-4 font-bold text-[#782819] flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-[#E78C79]" />
                          Negative (0)
                        </td>
                        <td className="py-3 px-4 text-right">80.56%</td>
                        <td className="py-3 px-4 text-right">79.12%</td>
                        <td className="py-3 px-4 text-right">79.83%</td>
                        <td className="py-3 px-4 text-right font-medium text-[#7E6D62]">10,000</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-bold text-[#1E562F] flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-[#96B8A2]" />
                          Positive (4)
                        </td>
                        <td className="py-3 px-4 text-right">79.15%</td>
                        <td className="py-3 px-4 text-right">80.56%</td>
                        <td className="py-3 px-4 text-right">79.85%</td>
                        <td className="py-3 px-4 text-right font-medium text-[#7E6D62]">10,000</td>
                      </tr>
                      <tr className="bg-[#FAF5EF]/60 font-bold text-[#2D231E]">
                        <td className="py-3 px-4">Macro Average</td>
                        <td className="py-3 px-4 text-right">79.85%</td>
                        <td className="py-3 px-4 text-right">79.84%</td>
                        <td className="py-3 px-4 text-right">79.84%</td>
                        <td className="py-3 px-4 text-right font-medium text-[#7E6D62]">20,000</td>
                      </tr>
                      <tr className="bg-[#FAF5EF]/60 font-bold text-[#2D231E]">
                        <td className="py-3 px-4">Weighted Average</td>
                        <td className="py-3 px-4 text-right">79.85%</td>
                        <td className="py-3 px-4 text-right">79.84%</td>
                        <td className="py-3 px-4 text-right">79.84%</td>
                        <td className="py-3 px-4 text-right font-medium text-[#7E6D62]">20,000</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Decision Threshold Sensitivity Card (5 cols) */}
              <div className="lg:col-span-5 retro-card p-6 flex flex-col justify-between h-full gap-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-[#2D231E] uppercase tracking-wider flex items-center gap-2">
                    <SlidersHorizontal className="w-4 h-4 text-[#8DA4EF]" />
                    Decision Threshold Sensitivity
                  </h3>
                  <span className="text-xs font-bold text-[#422C79] bg-[#EDE9FE] px-2.5 py-1 rounded-lg border border-[#DDD6FE] tabular-nums">
                    τ = {simThreshold.toFixed(2)}
                  </span>
                </div>

                {/* Interactive Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-bold text-[#7E6D62]">
                    <span>0.10 (High Recall)</span>
                    <span className="text-[#2D231E] tabular-nums">τ = {simThreshold.toFixed(2)}</span>
                    <span>0.90 (High Precision)</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={simThreshold}
                    onChange={(e) => setSimThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#8DA4EF] cursor-pointer h-2 bg-gray-200 rounded-lg"
                  />
                </div>

                {/* Sensitivity trade-off stats */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E5E0D8]">
                    <span className="text-xs font-bold uppercase tracking-wider text-[#7E6D62] block leading-none">Estimated Recall</span>
                    <p className="text-xl font-black text-[#1E562F] mt-2 tabular-nums leading-none">
                      {(Math.min(0.95, Math.max(0.60, 0.7912 + (0.50 - simThreshold) * 0.40)) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-[#FAF8F5] border border-[#E5E0D8]">
                    <span className="text-xs font-bold uppercase tracking-wider text-[#7E6D62] block leading-none">Estimated Precision</span>
                    <p className="text-xl font-black text-[#422C79] mt-2 tabular-nums leading-none">
                      {(Math.min(0.95, Math.max(0.60, 0.8029 + (simThreshold - 0.50) * 0.35)) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <p className="text-xs text-[#7E6D62] leading-relaxed italic">
                  Tuning τ allows trading off false positives against false negatives based on operational requirements.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ─── Warm Sage Footer matching Inspiration Image ─── */}
      <footer className="w-full bg-[#CADBD1] border-t border-[#B3C7BC] py-4">
        <div className="max-w-[1400px] mx-auto px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#2D231E]">
          <div className="flex flex-wrap items-center gap-2 font-semibold text-[#2D231E]">
            <span className="font-extrabold text-[#1F2C23]">Sentiment140</span>
            <span className="text-[#8FA89A]">•</span>
            <span className="text-[#3F4F44]">Artificial Intelligence Principles Project 11</span>
            <span className="text-[#8FA89A]">•</span>
            <span className="text-[#3F4F44]">Mohit, Vyom, Brian</span>
          </div>
          <div className="flex items-center gap-2 font-semibold text-[#293A2F]">
            <span className="px-3 py-1 rounded-md bg-[#BDD1C6] border border-[#AEC5B8]">React + Vite</span>
            <span className="px-3 py-1 rounded-md bg-[#BDD1C6] border border-[#AEC5B8]">Tailwind CSS</span>
            <span className="px-3 py-1 rounded-md bg-[#BDD1C6] border border-[#AEC5B8]">FastAPI</span>
            <span className="px-3 py-1 rounded-md bg-[#BDD1C6] border border-[#AEC5B8]">Scikit-Learn</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
