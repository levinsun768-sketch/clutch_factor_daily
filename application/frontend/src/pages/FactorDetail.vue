<script setup lang="ts">
import { computed, inject, onMounted, ref, watch } from 'vue';
import MiniSparkline from '../components/MiniSparkline.vue';
import LayerBacktestChart from '../components/LayerBacktestChart.vue';
import StyleBars from '../components/StyleBars.vue';
import { getJson, query } from '../services/api';
const props = defineProps<{ factorId: string }>();
const context = inject<any>('globalContext');
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const data = ref<any>(null);
const statusItems = computed(() => [
  { label: 'Predictive', value: data.value?.status?.predictive || '-' },
  { label: 'Stable', value: data.value?.status?.stable || '-' },
  { label: 'Risk', value: data.value?.status?.risk || '-' },
]);
async function load() { data.value = await getJson(`/api/factors/${props.factorId}/summary${query({ date: context.date, universe: context.universe })}`); }
onMounted(load);
watch(() => [props.factorId, context.date, context.universe], load);
</script>

<template>
  <section class="page-grid factor-detail">
    <div class="section-head terminal-head">
      <div>
        <span class="eyebrow">Factor Dossier</span>
        <h1>{{ data?.factor_id || factorId }}</h1>
      </div>
      <div class="factor-title-meta">
        <span class="badge">{{ data?.nickname || '-' }}</span>
        <span class="date-chip">{{ context.universe.toUpperCase() }}</span>
      </div>
    </div>

    <section class="factor-hero panel">
      <div class="factor-hero-main">
        <span class="eyebrow">Signal Quality</span>
        <h2>{{ data?.nickname || t('factors.mainStyle') }}</h2>
        <div class="status-strip">
          <div v-for="item in statusItems" :key="item.label">
            <i :class="['status-dot', item.value]"></i>
            <span>{{ item.label }}</span>
            <strong>{{ item.value }}</strong>
          </div>
        </div>
      </div>
      <div class="metric-grid compact-metrics">
        <div class="metric-card"><span>{{ t('factors.rankIc') }}</span><strong>{{ data?.rank_ic?.toFixed?.(4) ?? '-' }}</strong></div>
        <div class="metric-card"><span>{{ t('factors.icir') }}</span><strong>{{ data?.icir?.toFixed?.(2) ?? '-' }}</strong></div>
        <div class="metric-card"><span>{{ t('factors.netRet') }}</span><strong>{{ data?.net_cumret !== null && data?.net_cumret !== undefined ? `${(data.net_cumret * 100).toFixed(1)}%` : '-' }}</strong></div>
        <div class="metric-card"><span>{{ t('factors.turnover') }}</span><strong>{{ data?.turnover !== null && data?.turnover !== undefined ? `${(data.turnover * 100).toFixed(1)}%` : '-' }}</strong></div>
      </div>
    </section>

    <div class="detail-layout">
      <section class="panel">
        <div class="panel-heading"><h2>{{ t('factors.icTrend') }}</h2><small>{{ t('factors.latestIc') }}</small></div>
        <MiniSparkline :points="data?.ic_timeseries || []" :stroke-width="1.05" :height="90" :show-value="true" :label="t('factors.latestIc')" />
      </section>
      <section class="panel">
        <div class="panel-heading"><h2>{{ t('factors.styleExposure') }}</h2><small>Barra</small></div>
        <StyleBars :styles="data?.exposure" />
      </section>
    </div>

    <section class="panel">
      <div class="panel-heading"><h2>{{ t('factors.layeredBacktest') }}</h2><small>G1-G10 + Long-short</small></div>
      <LayerBacktestChart :backtest="data?.layer_backtest" />
    </section>

    <section class="panel">
      <div class="panel-heading"><h2>{{ t('common.recommendations') }}</h2><small>top exposure names</small></div>
      <table class="terminal-table"><tbody><tr v-for="row in data?.recommendations || []" :key="row.ts_code"><td>#{{ row.rank }}</td><td>{{ row.ts_code }}</td><td>{{ row.name }}</td><td>{{ row.industry }}</td><td>{{ row.score.toFixed(3) }}</td></tr></tbody></table>
    </section>
  </section>
</template>
