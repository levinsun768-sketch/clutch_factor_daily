<script setup lang="ts">
import { computed, inject, onMounted, ref, watch } from 'vue';
import MiniSparkline from '../components/MiniSparkline.vue';
import { getJson, query } from '../services/api';
const context = inject<any>('globalContext');
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const today = ref<any>(null);
const backtest = ref<any>(null);
const summary = computed(() => backtest.value?.summary || {});
const dailyRows = computed(() => backtest.value?.daily || []);
const navPoints = computed(() => dailyRows.value.map((x:any) => ({ date: x.date, value: x.net_nav })));
const turnoverPoints = computed(() => dailyRows.value.map((x:any) => ({ date: x.date, value: x.turnover })));
const drawdownPoints = computed(() => {
  let peak = 0;
  return dailyRows.value.map((x:any) => {
    const nav = Number(x.net_nav || 0);
    peak = Math.max(peak, nav);
    return { date: x.date, value: peak > 0 ? nav / peak - 1 : 0 };
  });
});
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
    <div class="portfolio-chart-grid">
      <section class="panel"><div class="panel-heading"><h2>{{ t('portfolio.nav') }}</h2><small>{{ t('portfolio.latestNav') }}</small></div><MiniSparkline :points="navPoints" :stroke-width="1.05" :height="82" :show-value="true" :label="t('portfolio.latestNav')" /></section>
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
    <section class="panel">
      <div class="panel-heading"><h2>{{ t('portfolio.holdings') }}</h2><small>{{ today?.holdings?.length || 0 }} names</small></div>
      <table class="terminal-table"><tbody><tr v-for="row in today?.holdings || []" :key="row.ts_code"><td>{{ row.ts_code }}</td><td>{{ row.name }}</td><td>{{ row.industry }}</td><td>#{{ row.rank }}</td><td>{{ row.composite_score.toFixed(3) }}</td></tr></tbody></table>
    </section>
  </section>
</template>
