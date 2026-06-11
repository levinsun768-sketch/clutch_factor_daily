<script setup lang="ts">
import { computed, inject, onMounted, ref, watch } from 'vue';
import MiniSparkline from '../components/MiniSparkline.vue';
import MultiLineChart from '../components/MultiLineChart.vue';
import { getJson, query } from '../services/api';
const context = inject<any>('globalContext');
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const today = ref<any>(null);
const backtest = ref<any>(null);
const showPortfolio = ref<boolean>(true);
const showBenchmark = ref<boolean>(true);
const showExcess = ref<boolean>(true);
const selectedBenchmark = ref<string>('hs300');
const summary = computed(() => backtest.value?.summary || {});
const dailyRows = computed(() => backtest.value?.daily || []);
const navSeries = computed(() => {
  const series = [];
  if (showPortfolio.value) {
    series.push({
      name: 'Portfolio',
      color: '#0ea5e9',
      data: dailyRows.value.map((x: any) => ({ date: x.date, value: x.net_nav }))
    });
  }
  if (showBenchmark.value) {
    series.push({
      name: 'Benchmark',
      color: '#6b7280',
      data: dailyRows.value.map((x: any) => ({ date: x.date, value: x.benchmark_nav || 1.0 }))
    });
  }
  if (showExcess.value) {
    series.push({
      name: 'Excess',
      color: '#10b981',
      data: dailyRows.value.map((x: any) => ({ date: x.date, value: x.excess_nav || 1.0 }))
    });
  }
  return series;
});
const turnoverPoints = computed(() => dailyRows.value.map((x:any) => ({ date: x.date, value: x.turnover })));
const drawdownPoints = computed(() => {
  let peak = 0;
  return dailyRows.value.map((x:any) => {
    const nav = Number(x.net_nav || 0);
    peak = Math.max(peak, nav);
    return { date: x.date, value: peak > 0 ? nav / peak - 1 : 0 };
  });
});
const styleTimeseries = computed(() => backtest.value?.style_timeseries || []);
const styleFactors = ['size', 'value_bp', 'momentum_252_20', 'reversal_20', 'beta_120', 'volatility_60', 'liquidity_amount_20'];
function getStyleTimeseries(styleName: string) {
  return styleTimeseries.value.map((x: any) => ({ date: x.date, value: x[styleName] || 0 }));
}
function getStyleColor(styleName: string): string {
  const colors: Record<string, string> = {
    size: '#0ea5e9', value_bp: '#8b5cf6', momentum_252_20: '#10b981',
    reversal_20: '#f59e0b', beta_120: '#ef4444', volatility_60: '#ec4899',
    liquidity_amount_20: '#06b6d4'
  };
  return colors[styleName] || '#6b7280';
}
const factorWeights = computed(() => Object.entries(summary.value.factor_weights || {}).map(([key, value]) => `${key}: ${Number(value).toFixed(2)}`).join(' · '));
async function load() {
  today.value = await getJson(`/api/portfolio/today${query({ date: context.date, universe: context.universe })}`);
  backtest.value = await getJson(`/api/portfolio/backtest${query({ universe: context.universe })}`);
}
onMounted(load);
watch(() => [context.date, context.universe], load);
</script>

<template>
  <section class="page-grid">
    <div class="section-head terminal-head">
      <div><span class="eyebrow">Production Strategy</span><h1>{{ t('portfolio.title') }}</h1></div>
      <span class="date-chip">{{ today?.trade_date }}</span>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><span>{{ t('portfolio.finalNav') }}</span><strong>{{ summary?.final_nav?.toFixed?.(2) ?? '-' }}</strong></div>
      <div class="metric-card"><span>{{ t('portfolio.sharpe') }}</span><strong>{{ summary?.sharpe_daily_sqrt252?.toFixed?.(2) ?? '-' }}</strong></div>
      <div class="metric-card"><span>{{ t('portfolio.todayTurnover') }}</span><strong>{{ today?.return_row?.turnover !== null && today?.return_row?.turnover !== undefined ? `${(today.return_row.turnover * 100).toFixed(1)}%` : '-' }}</strong></div>
      <div class="metric-card"><span>{{ t('portfolio.avgTurnover') }}</span><strong>{{ summary?.avg_turnover !== null && summary?.avg_turnover !== undefined ? `${(summary.avg_turnover * 100).toFixed(1)}%` : '-' }}</strong></div>
    </div>
    <section class="panel nav-comparison-panel">
      <div class="panel-heading">
        <h2>{{ t('portfolio.nav') }}</h2>
        <div class="benchmark-badge">
          <span class="badge-label">Benchmark:</span>
          <span class="badge-value">HS300</span>
        </div>
        <div class="nav-toggles">
          <label class="toggle-item">
            <input type="checkbox" v-model="showPortfolio">
            <span :style="{color: showPortfolio ? '#0ea5e9' : '#666'}">{{ t('portfolio.portfolioNav') }}</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" v-model="showBenchmark">
            <span :style="{color: showBenchmark ? '#6b7280' : '#666'}">{{ t('portfolio.benchmarkNav') }}</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" v-model="showExcess">
            <span :style="{color: showExcess ? '#10b981' : '#666'}">{{ t('portfolio.excessNav') }}</span>
          </label>
        </div>
      </div>
      <MultiLineChart :series="navSeries" :height="200" :show-legend="true" />
    </section>

    <div class="portfolio-chart-grid">
      <section class="panel"><div class="panel-heading"><h2>Drawdown</h2><small>peak-to-date</small></div><MiniSparkline :points="drawdownPoints" :stroke-width="1.05" :height="82" :show-value="true" label="Max stress" /></section>
      <section class="panel"><div class="panel-heading"><h2>Turnover</h2><small>daily</small></div><MiniSparkline :points="turnoverPoints" :stroke-width="1.05" :height="82" :show-value="true" label="Latest" /></section>
    </div>
    <section class="panel strategy-details">
      <div class="panel-heading"><h2>{{ t('portfolio.strategyDetails') }}</h2><small>top10 / buffer20 / industry cap</small></div>
      <div class="detail-grid">
        <div><span>{{ t('portfolio.dateRange') }}</span><strong>{{ summary.start_date }} - {{ summary.end_date }}</strong></div>
        <div><span>{{ t('portfolio.topN') }}</span><strong>{{ summary.top_n }}</strong></div>
        <div><span>{{ t('portfolio.sellRank') }}</span><strong>Top {{ summary.sell_rank }}</strong></div>
        <div><span>{{ t('portfolio.costBps') }}</span><strong>{{ summary.cost_bps }} bps</strong></div>
        <div><span>{{ t('portfolio.maxIndustryCount') }}</span><strong>{{ summary.max_industry_count }}</strong></div>
        <div><span>{{ t('portfolio.liquidityFloor') }}</span><strong>{{ summary.liquidity_floor }}</strong></div>
        <div><span>{{ t('portfolio.maxVolatility') }}</span><strong>{{ summary.max_volatility }}</strong></div>
        <div><span>{{ t('portfolio.factorWeights') }}</span><strong>{{ factorWeights }}</strong></div>
      </div>
    </section>
    <section class="panel" v-if="styleTimeseries.length > 0">
      <div class="panel-heading"><h2>{{ t('portfolio.barraExposure') }}</h2><small>{{ styleTimeseries.length }} days</small></div>
      <div class="style-grid">
        <div v-for="style in styleFactors" :key="style" class="style-item">
          <h3>{{ t('styleFactors.' + style) }}</h3>
          <MiniSparkline :points="getStyleTimeseries(style)" :stroke-width="1.05" :height="60" :color="getStyleColor(style)" />
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-heading"><h2>{{ t('portfolio.holdings') }}</h2><small>{{ today?.holdings?.length || 0 }} names</small></div>
      <table class="terminal-table"><tbody><tr v-for="row in today?.holdings || []" :key="row.ts_code"><td>{{ row.ts_code }}</td><td>{{ row.name }}</td><td>{{ row.industry }}</td><td>#{{ row.rank }}</td><td>{{ row.composite_score.toFixed(3) }}</td></tr></tbody></table>
    </section>
  </section>
</template>

<style scoped>
.head-controls { display: flex; align-items: center; gap: 1rem; }
.benchmark-select {
  padding: 0.375rem 0.75rem;
  font-size: 0.875rem;
  background: #1a1a1a;
  color: #fff;
  border: 1px solid #333;
  border-radius: 4px;
  cursor: pointer;
  outline: none;
}
.benchmark-select:hover { border-color: #0ea5e9; }
.benchmark-select:focus { border-color: #0ea5e9; }
.nav-comparison-panel { grid-column: 1 / -1; }
.benchmark-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  background: rgba(107, 114, 128, 0.1);
  border: 1px solid #6b7280;
  border-radius: 4px;
  margin: 0 auto 0 1rem;
}
.badge-label {
  font-size: 0.75rem;
  color: #999;
}
.badge-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: #6b7280;
}
.nav-toggles { display: flex; gap: 1.5rem; margin-left: auto; }
.toggle-item { display: flex; align-items: center; gap: 0.5rem; cursor: pointer; font-size: 0.875rem; }
.toggle-item input[type="checkbox"] { width: 16px; height: 16px; cursor: pointer; }
.toggle-item span { transition: color 0.2s; }
.style-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; padding: 1rem; }
.style-item { border: 1px solid #333; border-radius: 4px; padding: 0.75rem; }
.style-item h3 { font-size: 0.875rem; margin: 0 0 0.5rem 0; color: #fff; }
</style>
