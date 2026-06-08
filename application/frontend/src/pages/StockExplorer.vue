<script setup lang="ts">
import { inject, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { Search } from '@lucide/vue';
import StyleBars from '../components/StyleBars.vue';
import { getJson, query } from '../services/api';
const props = defineProps<{ tsCode?: string }>();
const router = useRouter();
const context = inject<any>('globalContext');
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const code = ref(props.tsCode || '');
const profile = ref<any>(null);
const similar = ref<any[]>([]);
const error = ref('');
async function load() {
  if (!code.value) return;
  error.value = '';
  try {
    profile.value = await getJson(`/api/stocks/${code.value}/profile${query({ date: context.date, universe: context.universe })}`);
    const res = await getJson<any>(`/api/stocks/${code.value}/similar${query({ date: context.date, universe: context.universe, top_n: 20 })}`);
    similar.value = res.items || [];
  } catch (err) {
    error.value = String(err);
  }
}
function open() { if (!code.value.trim()) return; router.push(`/stocks/${code.value.trim()}`); load(); }
watch(() => [props.tsCode, context.date, context.universe], () => { code.value = props.tsCode || code.value; load(); }, { immediate: true });
</script>

<template>
  <section class="page-grid stock-explorer">
    <div class="section-head terminal-head">
      <div><span class="eyebrow">Stock Lens</span><h1>{{ t('stocks.title') }}</h1></div>
      <div class="search-box stock-search">
        <input v-model="code" placeholder="600519.SH" @keydown.enter="open" />
        <button class="icon-text-button" @click="open"><Search :size="15" />{{ t('common.open') }}</button>
      </div>
    </div>
    <div v-if="error" class="notice">{{ error }}</div>

    <section class="stock-hero panel">
      <div class="stock-identity">
        <span class="eyebrow">{{ profile?.ts_code || code || '-' }}</span>
        <h2>{{ profile?.name ?? '-' }}</h2>
        <p>{{ profile?.industry ?? '-' }}</p>
      </div>
      <div class="metric-grid compact-metrics">
        <div class="metric-card"><span>{{ t('common.composite') }}</span><strong>{{ profile?.composite_score?.toFixed?.(3) ?? '-' }}</strong></div>
        <div class="metric-card"><span>{{ t('common.industry') }}</span><strong>{{ profile?.industry ?? '-' }}</strong></div>
        <div class="metric-card"><span>{{ t('controls.universe') }}</span><strong>{{ context.universe.toUpperCase() }}</strong></div>
      </div>
    </section>

    <div class="detail-layout">
      <section class="panel">
        <div class="panel-heading"><h2>{{ t('factors.styleExposure') }}</h2><small>Barra</small></div>
        <StyleBars :styles="profile?.style_exposure" />
      </section>
      <section class="panel">
        <div class="panel-heading"><h2>{{ t('stocks.similarStocks') }}</h2><small>fingerprint cosine</small></div>
        <table class="terminal-table"><tbody><tr v-for="row in similar" :key="row.ts_code"><td>#{{ row.rank }}</td><td>{{ row.ts_code }}</td><td>{{ row.name }}</td><td>{{ row.industry }}</td><td>{{ row.similarity.toFixed(3) }}</td></tr></tbody></table>
      </section>
    </div>
  </section>
</template>
