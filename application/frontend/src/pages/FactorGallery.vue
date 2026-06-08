<script setup lang="ts">
import { inject, onMounted, ref, watch } from 'vue';
import MiniSparkline from '../components/MiniSparkline.vue';
import { getJson, query } from '../services/api';

const context = inject<any>('globalContext');
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const factors = ref<any[]>([]);
const sort = ref('rankic');
const style = ref('all');
const sortOptions = ['rankic', 'icir', 'return', 'turnover', 'style'];
const styleOptions = ['all', 'Momentum', 'Reversal', 'Size', 'Value', 'Volatility', 'Liquidity', 'Beta', 'Neutral Alpha'];
async function load() {
  const res = await getJson<any>(`/api/factors${query({ date: context.date, universe: context.universe, sort: sort.value, style: style.value })}`);
  factors.value = res.items || [];
}
onMounted(load);
watch(() => [context.date, context.universe, sort.value, style.value], load);
</script>

<template>
  <section class="page-grid">
    <div class="section-head terminal-head">
      <div><span class="eyebrow">Factor Radar</span><h1>{{ t('factors.gallery') }}</h1></div>
      <span class="date-chip">{{ factors.length }} factors</span>
    </div>
    <div class="factor-toolbar panel">
      <div class="segmented"><button v-for="s in sortOptions" :key="s" :class="{ active: sort === s }" @click="sort = s">{{ t(`factors.sort.${s}`) }}</button></div>
      <div class="filter-row"><button v-for="s in styleOptions" :key="s" :class="{ active: style === s }" @click="style = s">{{ t(`factors.styles.${s}`) }}</button></div>
    </div>
    <div class="factor-list">
      <RouterLink v-for="item in factors" :key="item.factor_id" class="factor-card" :to="`/factors/${item.factor_id}`">
        <div class="factor-main">
          <span class="factor-id">{{ item.factor_id }}</span>
          <strong>{{ item.nickname }}</strong>
          <span class="badge">{{ t(`factors.styles.${item.main_style || 'Unknown'}`) }}</span>
        </div>
        <div class="factor-metrics">
          <span>{{ t('factors.rankIc') }} <b>{{ item.rank_ic?.toFixed?.(4) ?? '-' }}</b></span>
          <span>{{ t('factors.icir') }} <b>{{ item.icir?.toFixed?.(2) ?? '-' }}</b></span>
          <span>{{ t('factors.turnover') }} <b>{{ item.turnover !== null && item.turnover !== undefined ? `${(item.turnover * 100).toFixed(1)}%` : t('common.notAvailable') }}</b></span>
          <span>{{ t('factors.ret') }} <b>{{ item.net_cumret !== null && item.net_cumret !== undefined ? `${(item.net_cumret * 100).toFixed(1)}%` : t('common.notAvailable') }}</b></span>
        </div>
        <div class="factor-signal">
          <div class="status-dots" :title="`predictive ${item.status?.predictive || '-'} / stable ${item.status?.stable || '-'} / risk ${item.status?.risk || '-'}`">
            <i :class="['status-dot', item.status?.predictive || '']"></i>
            <i :class="['status-dot', item.status?.stable || '']"></i>
            <i :class="['status-dot', item.status?.risk || '']"></i>
          </div>
          <MiniSparkline :points="item.sparkline || []" />
        </div>
      </RouterLink>
    </div>
  </section>
</template>
