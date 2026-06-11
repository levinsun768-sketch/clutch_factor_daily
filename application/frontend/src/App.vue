<script setup lang="ts">
import { computed, provide, reactive, watch } from 'vue';
import { useRoute } from 'vue-router';
import { BarChart3, Briefcase, GalleryVerticalEnd, Home, Search } from '@lucide/vue';
import GlobalControls from './components/GlobalControls.vue';
import { initialLocale, translate, type Locale } from './i18n';

const route = useRoute();
const context = reactive({
  date: '',
  universe: 'all',
  locale: initialLocale(),
});
const t = (key: string) => translate(context.locale as Locale, key);
provide('globalContext', context);
provide('i18n', { t });
watch(() => context.locale, (locale) => localStorage.setItem('locale', locale));

const navItems = [
  { path: '/', labelKey: 'nav.home', icon: Home },
  { path: '/factors', labelKey: 'nav.factors', icon: GalleryVerticalEnd },
  { path: '/stocks', labelKey: 'nav.stocks', icon: Search },
  { path: '/portfolio', labelKey: 'nav.portfolio', icon: Briefcase },
];
</script>

<template>
  <div class="app-shell">
    <aside class="side-nav">
      <div class="brand">
        <div class="brand-mark"><BarChart3 :size="20" /></div>
        <div class="brand-copy"><strong>Clutch Factor</strong><small>Research Terminal</small></div>
      </div>
      <div class="nav-section">Workspace</div>
      <RouterLink v-for="item in navItems" :key="item.path" :to="item.path" class="nav-item">
        <component :is="item.icon" :size="18" />
        <span>{{ t(item.labelKey) }}</span>
      </RouterLink>
      <div class="nav-footer">
        <small>Product layer synced</small>
      </div>
    </aside>
    <main class="workspace">
      <GlobalControls v-model:date="context.date" v-model:universe="context.universe" v-model:locale="context.locale" />
      <RouterView />
    </main>
  </div>
</template>
