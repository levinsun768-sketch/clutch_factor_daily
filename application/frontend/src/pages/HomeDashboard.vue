<script setup lang="ts">
import { inject, onMounted, ref, watch } from 'vue';
import { getJson, query } from '../services/api';
import StyleBars from '../components/StyleBars.vue';
import IndustryBars from '../components/IndustryBars.vue';

const context = inject<any>('globalContext');
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const data = ref<any>(null);
const error = ref('');
async function load() {
  error.value = '';
  try {
    data.value = await getJson(`/api/overview${query({ date: context.date, universe: context.universe })}`);
  } catch (err) {
    error.value = String(err);
  }
}
onMounted(load);
watch(() => [context.date, context.universe], load);
</script>

<template>
  <section class="page-grid">
    <div class="section-head terminal-head">
      <div><span class="eyebrow">Market Surface</span><h1>{{ t('home.title') }}</h1></div>
      <span class="date-chip">{{ data?.date || t('common.latest') }}</span>
    </div>
    <div v-if="error" class="notice">{{ error }}</div>
    <div class="market-tape">
      <div class="metric-card tape-card"><span>{{ t('home.universeSize') }}</span><strong>{{ data?.market?.n ?? '-' }}</strong></div>
      <div class="metric-card tape-card"><span>{{ t('home.upDown') }}</span><strong><b class="positive">{{ data?.market?.up ?? '-' }}</b> / <b class="negative">{{ data?.market?.down ?? '-' }}</b></strong></div>
      <div class="metric-card tape-card"><span>{{ t('home.limitUp') }}</span><strong>{{ data?.market?.limit_up ?? '-' }}</strong></div>
      <div class="metric-card tape-card"><span>{{ t('home.avgPctChg') }}</span><strong :class="{ positive: data?.market?.avg_pct_chg > 0, negative: data?.market?.avg_pct_chg < 0 }">{{ data?.market?.avg_pct_chg !== null && data?.market?.avg_pct_chg !== undefined ? `${data.market.avg_pct_chg.toFixed(2)}%` : '-' }}</strong></div>
    </div>
    <section class="panel panel-emphasis">
      <div class="panel-heading"><h2>{{ t('home.styleMonitor') }}</h2><small>{{ data?.style_monitor?.unit || '' }}</small></div>
      <StyleBars :styles="data?.style_monitor?.styles" :unit="data?.style_monitor?.unit" />
    </section>
    <section class="panel panel-emphasis">
      <div class="panel-heading"><h2>{{ t('home.industryMonitor') }}</h2><small>{{ t('common.relativePremium') }}</small></div>
      <IndustryBars :industries="data?.style_monitor?.industry_premium" />
    </section>
  </section>
</template>
