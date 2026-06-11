<script setup lang="ts">
import { computed, ref, onMounted, watch } from 'vue';

interface DataPoint {
  date: string;
  value: number;
}

interface Series {
  name: string;
  color: string;
  data: DataPoint[];
}

const props = defineProps<{
  series: Series[];
  height?: number;
  showLegend?: boolean;
}>();

const container = ref<HTMLElement | null>(null);
const width = ref(800);
const height = props.height || 200;
const padding = { top: 20, right: 20, bottom: 30, left: 50 };

const allValues = computed(() => {
  return props.series.flatMap(s => s.data.map(d => d.value));
});

const yMin = computed(() => Math.min(...allValues.value) * 0.95);
const yMax = computed(() => Math.max(...allValues.value) * 1.05);

function scaleY(value: number): number {
  const range = yMax.value - yMin.value;
  if (range === 0) return height / 2;
  return height - padding.bottom - ((value - yMin.value) / range) * (height - padding.top - padding.bottom);
}

function scaleX(index: number, total: number): number {
  const chartWidth = width.value - padding.left - padding.right;
  return padding.left + (index / (total - 1 || 1)) * chartWidth;
}

function buildPath(data: DataPoint[]): string {
  if (data.length === 0) return '';
  const total = data.length;
  let path = '';
  data.forEach((point, i) => {
    const x = scaleX(i, total);
    const y = scaleY(point.value);
    path += (i === 0 ? 'M' : 'L') + `${x},${y}`;
  });
  return path;
}

const yTicks = computed(() => {
  const ticks = [];
  const step = (yMax.value - yMin.value) / 4;
  for (let i = 0; i <= 4; i++) {
    const value = yMin.value + step * i;
    ticks.push({ value, y: scaleY(value) });
  }
  return ticks;
});

onMounted(() => {
  if (container.value) {
    width.value = container.value.clientWidth;
  }
});
</script>

<template>
  <div ref="container" class="multi-line-chart">
    <svg :width="width" :height="height" class="chart-svg">
      <!-- Grid lines -->
      <g class="grid">
        <line v-for="tick in yTicks" :key="tick.value"
          :x1="padding.left" :y1="tick.y"
          :x2="width - padding.right" :y2="tick.y"
          stroke="#333" stroke-width="1" stroke-dasharray="2,2" />
      </g>

      <!-- Y axis labels -->
      <g class="y-axis">
        <text v-for="tick in yTicks" :key="tick.value"
          :x="padding.left - 10" :y="tick.y + 4"
          text-anchor="end" font-size="11" fill="#999">
          {{ tick.value.toFixed(2) }}
        </text>
      </g>

      <!-- Lines -->
      <g class="lines">
        <path v-for="s in series" :key="s.name"
          :d="buildPath(s.data)"
          :stroke="s.color"
          stroke-width="2"
          fill="none"
          stroke-linejoin="round"
          stroke-linecap="round" />
      </g>
    </svg>

    <!-- Legend -->
    <div v-if="showLegend !== false" class="legend">
      <div v-for="s in series" :key="s.name" class="legend-item">
        <span class="legend-color" :style="{ backgroundColor: s.color }"></span>
        <span class="legend-label">{{ s.name }}</span>
        <span class="legend-value">{{ s.data[s.data.length - 1]?.value.toFixed(2) }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.multi-line-chart {
  width: 100%;
  position: relative;
}

.chart-svg {
  display: block;
}

.legend {
  display: flex;
  gap: 1.5rem;
  margin-top: 0.75rem;
  padding: 0 0.5rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.legend-label {
  color: #999;
}

.legend-value {
  color: #fff;
  font-weight: 500;
}
</style>
