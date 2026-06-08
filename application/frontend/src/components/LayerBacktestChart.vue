<script setup lang="ts">
import { computed, inject } from 'vue';

const props = defineProps<{ backtest?: any }>();
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const groups = Array.from({ length: 10 }, (_, i) => i + 1);

function cleanSeries(rows: any[], key: string) {
  return rows.map((row) => ({ date: row.date, value: Number(row[key]) })).filter((row) => Number.isFinite(row.value));
}

function pathFor(points: Array<{ date: string; value: number }>, min: number, max: number) {
  if (points.length < 2) return '';
  const span = Math.max(max - min, 1e-9);
  return points.map((point, index) => `${(index / (points.length - 1)) * 100},${100 - ((point.value - min) / span) * 100}`).join(' ');
}

const groupRows = computed(() => props.backtest?.group_nav || []);
const longShortRows = computed(() => props.backtest?.long_short || []);
const groupSeries = computed(() => groups.map((group) => ({
  group,
  points: cleanSeries(groupRows.value, `group_${group}`),
})));
const longShort = computed(() => cleanSeries(longShortRows.value, 'net_nav'));
const yRange = computed(() => {
  const values = [
    ...groupSeries.value.flatMap((series) => series.points.map((point) => point.value)),
    ...longShort.value.map((point) => point.value),
  ];
  if (!values.length) return { min: 0, max: 1 };
  return { min: Math.min(...values), max: Math.max(...values) };
});
const paths = computed(() => groupSeries.value.map((series) => ({
  group: series.group,
  d: pathFor(series.points, yRange.value.min, yRange.value.max),
})));
const lsPath = computed(() => pathFor(longShort.value, yRange.value.min, yRange.value.max));
const latest = computed(() => {
  const summary = props.backtest?.summary || {};
  return {
    nav: summary.final_net_nav,
    ret: summary.net_cumret,
    turnover: summary.ls_turnover_mean,
    win: summary.net_win_rate,
  };
});
function pct(value: number | null | undefined) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '-';
}
function dec(value: number | null | undefined) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : '-';
}
</script>

<template>
  <div v-if="backtest?.available" class="layer-backtest">
    <div class="chart-header compact">
      <div><span>{{ t('factors.longShortNav') }}</span><strong>{{ dec(latest.nav) }}</strong></div>
      <div><span>{{ t('factors.netRet') }}</span><strong>{{ pct(latest.ret) }}</strong></div>
      <div><span>{{ t('factors.turnover') }}</span><strong>{{ pct(latest.turnover) }}</strong></div>
      <div><span>{{ t('factors.winRate') }}</span><strong>{{ pct(latest.win) }}</strong></div>
    </div>
    <svg class="layer-chart" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline v-for="item in paths" :key="item.group" :points="item.d" :class="`group-line group-${item.group}`" fill="none" vector-effect="non-scaling-stroke" />
      <polyline :points="lsPath" class="ls-line" fill="none" vector-effect="non-scaling-stroke" />
    </svg>
    <div class="chart-legend ten-groups">
      <span><i class="legend-ls"></i>Long-short</span>
      <span v-for="group in groups" :key="group"><i :class="`legend-group group-${group}`"></i>G{{ group }}</span>
    </div>
  </div>
  <div v-else class="empty-state">{{ t('factors.noLayerBacktest') }}</div>
</template>
