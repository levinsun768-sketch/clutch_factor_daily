import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import HomeDashboard from './pages/HomeDashboard.vue';
import FactorGallery from './pages/FactorGallery.vue';
import FactorDetail from './pages/FactorDetail.vue';
import StockExplorer from './pages/StockExplorer.vue';
import PortfolioBoard from './pages/PortfolioBoard.vue';
import './styles/main.css';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: HomeDashboard },
    { path: '/factors', component: FactorGallery },
    { path: '/factors/:factorId', component: FactorDetail, props: true },
    { path: '/stocks/:tsCode?', component: StockExplorer, props: true },
    { path: '/portfolio', component: PortfolioBoard }
  ]
});

createApp(App).use(router).mount('#app');
