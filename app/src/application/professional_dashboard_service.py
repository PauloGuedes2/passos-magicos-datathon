"""Enterprise dashboard service."""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from src.config.settings import Configuracoes
from src.util.logger import logger

try:
    from plotly.offline import get_plotlyjs
except Exception:  # pragma: no cover
    get_plotlyjs = None


class ProfessionalDashboardService:
    """Gera dashboard executivo de monitoramento a partir dos artefatos persistidos."""

    _PLOTLY_JS_CACHE: str | None = None

    def __init__(self, monitoring_dir: str | None = None, output_file: str | None = None):
        """Define caminhos de leitura/escrita para artefatos e HTML final."""
        self.monitoring_dir = monitoring_dir or Configuracoes.MONITORING_DIR
        self.output_file = output_file or os.path.join(self.monitoring_dir, "professional_dashboard.html")

    def generate_dashboard(self) -> str:
        """Gera o dashboard consolidado e retorna o caminho do arquivo produzido."""
        train = self._read_json("train_metrics.json")
        temporal = self._read_json("temporal_cv_report.json")
        drift = self._read_json("drift_report.json")
        cmp_data = self._read_json("threshold_strategy_comparison.json")
        active = self._resolve_active_strategy(train, cmp_data)
        rows = self._resolve_threshold_rows(train, cmp_data, active)
        ctx = {"train": train if isinstance(train, dict) else {}, "temporal": temporal if isinstance(temporal, dict) else {}, "drift": drift if isinstance(drift, dict) else {}, "rows": rows, "active": active, "generated_at": datetime.now(timezone.utc).isoformat(), "environment": os.getenv("APP_ENV", "producao")}
        ctx["health"] = self._model_health(ctx)
        payload = self._build_payload(ctx)
        sections = [self.build_operational_impact_section(ctx), self.build_performance_section(ctx), self.build_drift_section(ctx), self.build_explainability_section(ctx), self.build_governance_section(ctx), self.build_strategic_conclusion(ctx)]
        if payload.get("fairness", {}).get("data"):
            sections.insert(4, self.build_fairness_section(ctx))
        doc = self.assemble_dashboard(ctx, sections, payload)
        self._write_text(self.output_file, doc)
        logger.info(f"Dashboard profissional gerado em: {self.output_file}")
        return self.output_file

    def build_operational_impact_section(self, ctx: dict[str, Any]) -> str:
        """Constroi a secao de impacto operacional com risco e capacidade de revisao."""
        drift = ctx["drift"] if isinstance(ctx.get("drift"), dict) else {}
        strategic = drift.get("strategic_monitoring", {}) if isinstance(drift.get("strategic_monitoring"), dict) else {}
        current = self._f(strategic.get("current_high_risk_rate_pct"))
        delta = self._f(strategic.get("delta_high_risk_pp"))
        active_row = next((r for r in ctx.get("rows", []) if str(r.get("strategy")) == str(ctx.get("active"))), ctx.get("rows", [{}])[0] if ctx.get("rows") else {})
        reviews = self._num(active_row.get("human_reviews"), 0)
        trend = "estavel"
        history = drift.get("high_risk_rate_history", []) if isinstance(drift.get("high_risk_rate_history"), list) else []
        if len(history) >= 2:
            prev = self._f(history[-2].get("rate")) if isinstance(history[-2], dict) else None
            cur = self._f(history[-1].get("rate")) if isinstance(history[-1], dict) else None
            if prev is not None and cur is not None:
                trend = "alta" if cur > prev else "queda" if cur < prev else "estavel"
        cards = [
            ("Taxa ALTO_RISCO", f"{current:.1f}%" if current is not None else "n/d"),
            ("Variacao", f"{delta:+.1f} p.p." if delta is not None else "n/d"),
            ("Revisoes estimadas", reviews),
            ("Tendencia", trend),
        ]
        grid = "".join("<article class='metric-card'><div class='metric-label'>" + html.escape(k) + "</div><div class='metric-value'>" + html.escape(v) + "</div></article>" for k, v in cards)
        return "<section id='risk-segmentation' class='section'>" + self._section_title("Impacto Operacional", "Traduz o comportamento do modelo em impacto no dia a dia da operacao.") + "<div class='metrics-grid'>" + grid + "</div><p class='section-intro'>A taxa de ALTO_RISCO indica o percentual de casos criticos no periodo. Se subir rapidamente, a carga operacional aumenta e pode exigir priorizacao de equipe.</p>" + self._threshold_table(ctx["rows"], str(ctx.get("active", "n/d"))) + "<div id='threshold-strategy-compare' class='plot-card'></div><div class='plot-grid'><div id='high-risk-evolution' class='plot-card'></div><div id='risk-segmentation-chart' class='plot-card'></div></div></section>"

    def build_performance_section(self, ctx: dict[str, Any]) -> str:
        """Constroi a secao de performance com curvas e metricas de classificacao."""
        return "<section id='model-performance' class='section'>" + self._section_title("Performance do Modelo", "Indicadores centrais de qualidade de classificacao.") + "<p class='section-intro'>Curva ROC, curva Precisao-Recall, matriz de confusao, distribuicao de probabilidades e trade-off de limiar.</p><p class='section-intro'>Leitura executiva: curvas acima da diagonal e PR consistente sustentam separabilidade, enquanto o trade-off explicita o custo entre recall e falsos positivos.</p><div class='plot-grid'><div class='plot-stack'><div id='roc-curve-plot' class='plot-card'></div><div class='hint-card'>O que isso significa? Mostra a capacidade do modelo de diferenciar casos positivos e negativos. Quanto mais proxima do canto superior esquerdo, melhor a separacao.</div></div><div class='plot-stack'><div id='pr-curve-plot' class='plot-card'></div><div class='hint-card'>O que isso significa? Mostra o equilibrio entre identificar corretamente casos positivos e evitar falsos alarmes.</div></div></div><div class='plot-grid'><div class='plot-stack'><div id='confusion-matrix-heatmap' class='plot-card'></div><div class='hint-card'>O que isso significa? Resume quantas previsoes foram corretas e incorretas no limiar atual.</div></div><div class='plot-stack'><div id='probability-distribution-plot' class='plot-card'></div><div class='hint-card'>O que isso significa? Mostra como o modelo distribui as probabilidades entre classes.</div></div></div><div class='plot-stack'><div id='threshold-tradeoff-plot' class='plot-card'></div><div class='hint-card'>O que isso significa? Ilustra como mudancas no limiar impactam recall e precisao.</div></div></section>"

    def build_drift_section(self, ctx: dict[str, Any]) -> str:
        """Constroi a secao de drift com indicadores de estabilidade e PSI."""
        avg_psi = ctx.get("health", {}).get("avg_psi")
        avg_text = f"{avg_psi:.4f}" if isinstance(avg_psi, float) else "n/d"
        status = self._drift_status(ctx["drift"])["label_short"]
        return "<section id='drift-analysis' class='section'>" + self._section_title("Drift dos Dados", "Drift significa mudanca no perfil dos dados em relacao ao treino.") + "<div class='metrics-grid'><article class='metric-card'><div class='metric-label'>Status consolidado</div><div class='metric-value'>" + html.escape(status) + "</div></article><article class='metric-card'><div class='metric-label'>PSI medio</div><div class='metric-value'>" + avg_text + "</div></article></div><p class='section-intro'>Quando o drift cresce, o modelo pode perder estabilidade. Isso nao significa erro imediato, mas exige monitoramento e possivel ajuste operacional.</p>" + self._drift_table(ctx["drift"]) + "<div class='plot-grid'><div id='psi-feature-bar' class='plot-card'></div><div id='drift-distribution-top5' class='plot-card'></div></div><div id='avg-psi-evolution' class='plot-card'></div></section>"

    def build_fairness_section(self, _ctx: dict[str, Any]) -> str:
        """Constroi a secao de equidade com comparativos por grupo."""
        return "<section id='fairness' class='section'>" + self._section_title("Equidade", "Comparativo entre grupos para monitoramento de equidade operacional.") + "<p class='section-intro'>Comparativo de Recall, Precisao e F1 por grupo para leitura executiva de consistencia entre segmentos.</p><div id='fairness-group-chart' class='plot-card fairness-plot'></div></section>"

    def build_governance_section(self, ctx: dict[str, Any]) -> str:
        """Constroi a secao de governanca com trilha de auditoria e metadados."""
        status = self._drift_status(ctx["drift"])
        threshold = self._num(ctx["train"].get("risk_threshold"), 4)
        generated_at = str(ctx.get("generated_at", "n/d"))
        version_source = str(ctx["train"].get("train_date") or ctx["temporal"].get("generated_at") or generated_at)
        model_version = str(ctx["train"].get("model_version") or ctx["train"].get("model_name") or "n/d")
        environment = str(ctx.get("environment", "n/d"))
        owner = str(os.getenv("MODEL_OWNER", "Time de Model Risk"))
        icon_clock = "<svg viewBox='0 0 24 24' aria-hidden='true'><circle cx='12' cy='12' r='8' fill='none' stroke='currentColor' stroke-width='1.6'/><path d='M12 8v4l3 2' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg>"
        icon_tag = "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M4 8h9l7 4-7 4H4z' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linejoin='round'/></svg>"
        icon_layers = "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 4l8 4-8 4-8-4 8-4zM4 12l8 4 8-4M4 16l8 4 8-4' fill='none' stroke='currentColor' stroke-width='1.4' stroke-linejoin='round'/></svg>"
        icon_slider = "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M6 8h12M6 16h12M9 8v4M15 12v4' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg>"
        icon_shield = "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M12 4l7 3v5c0 4-2.8 6.8-7 8-4.2-1.2-7-4-7-8V7z' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linejoin='round'/></svg>"
        icon_user = "<svg viewBox='0 0 24 24' aria-hidden='true'><circle cx='12' cy='9' r='3' fill='none' stroke='currentColor' stroke-width='1.6'/><path d='M6 19c0-3 2.4-5 6-5s6 2 6 5' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg>"
        icon_server = "<svg viewBox='0 0 24 24' aria-hidden='true'><rect x='5' y='6' width='14' height='4' rx='1' fill='none' stroke='currentColor' stroke-width='1.6'/><rect x='5' y='14' width='14' height='4' rx='1' fill='none' stroke='currentColor' stroke-width='1.6'/></svg>"
        return (
            "<section id='governance' class='section audit-section'>"
            + self._section_title("Governanca", "Barra de auditoria com metadados operacionais e tecnicos do ciclo atual.")
            + "<p class='section-intro'>Barra de auditoria para rastreabilidade e controle de versao.</p>"
            + "<div class='audit-bar' data-governance data-generated-at='"
            + html.escape(generated_at)
            + "' data-version-source='"
            + html.escape(version_source)
            + "'>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_clock
            + "</span><div class='audit-meta'><span class='audit-label'>Gerado em (SP)</span><span class='audit-value technical' data-governance-generated>"
            + html.escape(generated_at)
            + "</span></div></article>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_tag
            + "</span><div class='audit-meta'><span class='audit-label'>Versao do modelo</span><span class='audit-value technical' data-governance-version>"
            + html.escape(model_version)
            + "</span></div></article>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_layers
            + "</span><div class='audit-meta'><span class='audit-label'>Estrategia ativa</span><span class='audit-value technical'>"
            + html.escape(str(ctx.get("active", "n/d")))
            + "</span></div></article>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_slider
            + "</span><div class='audit-meta'><span class='audit-label'>Limiar <span class='info-tip'><span class='info-icon'>i</span><span class='tip-bubble'>Ponto de corte usado para classificar ALTO_RISCO.</span></span></span><span class='audit-value technical'>"
            + html.escape(threshold)
            + "</span></div></article>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_shield
            + "</span><div class='audit-meta'><span class='audit-label'>Status de drift</span><span class='audit-value'>"
            + html.escape(status["label_short"])
            + "</span></div></article>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_user
            + "</span><div class='audit-meta'><span class='audit-label'>Responsavel</span><span class='audit-value'>"
            + html.escape(owner)
            + "</span></div></article>"
            + "<article class='audit-item'><span class='audit-icon' aria-hidden='true'>"
            + icon_server
            + "</span><div class='audit-meta'><span class='audit-label'>Ambiente</span><span class='audit-value technical'>"
            + html.escape(environment)
            + "</span></div></article>"
            + "</div></section>"
        )

    def build_explainability_section(self, _ctx: dict[str, Any]) -> str:
        """Constroi a secao de explicabilidade com importancia global de features."""
        return "<section id='explainability' class='section'>" + self._section_title("Explicabilidade", "Importancia nao indica causalidade, apenas influencia no modelo.") + "<p class='section-intro'>Importancia global das variaveis para decisao executiva.</p><p class='section-intro'>As variaveis abaixo sao os principais fatores utilizados pelo modelo para estimar risco.</p><div id='feature-importance-global' class='plot-card explainability-plot'></div></section>"

    def build_strategic_conclusion(self, ctx: dict[str, Any]) -> str:
        """Constroi o resumo executivo com veredito, insights e proximos passos."""
        conclusions = ctx["temporal"].get("interpretacao", {}).get("conclusoes", [])
        if not isinstance(conclusions, list) or not conclusions:
            conclusions = ["Sem conclusoes temporais persistidas; manter monitoramento continuo."]
        health = ctx.get("health", {})
        score = self._f(health.get("score")) or 0.0
        drift_status = self._drift_status(ctx["drift"])["label_short"]
        verdict = "Aprovado"
        verdict_reason = "Modelo estavel e com risco operacional controlado."
        verdict_css = "verdict-approved"
        if drift_status.lower() in ("moderado",) or score < 75:
            verdict = "Atencao"
            verdict_reason = "Manter vigilancia ativa para evitar degradacao."
            verdict_css = "verdict-attention"
        if drift_status.lower() in ("critico",) or score < 60:
            verdict = "Reprovado"
            verdict_reason = "Necessita plano corretivo e revisao imediata."
            verdict_css = "verdict-rejected"
        actions = [
            "Priorizar revisao de dados quando PSI ultrapassar 0.25 em features criticas.",
            "Reavaliar threshold se a taxa de ALTO_RISCO variar de forma abrupta por periodos consecutivos.",
            "Manter monitoramento continuo com foco em estabilidade temporal e capacidade operacional.",
        ]
        highlights = conclusions[:3]
        highlights_html = "".join(f"<li>{html.escape(str(i))}</li>" for i in highlights)
        actions_html = "".join(f"<li>{html.escape(str(i))}</li>" for i in actions)
        return (
            "<section id='conclusion' class='section executive-summary' data-health-score='"
            + f"{score:.1f}"
            + "' data-drift-status='"
            + html.escape(drift_status)
            + "'>"
            + self._section_title("Conclusao e Recomendacao", "Resumo executivo para decisao rapida.")
            + "<p class='section-intro'>Resumo objetivo dos sinais de saude, riscos e proximas acoes.</p>"
            + "<div class='exec-grid'>"
            + "<article class='exec-card verdict-card'><span class='exec-kicker'>Veredito</span><div class='verdict-badge "
            + verdict_css
            + "' data-verdict-badge>"
            + html.escape(verdict)
            + "</div><p class='verdict-reason' data-verdict-reason>"
            + html.escape(verdict_reason)
            + "</p><small class='exec-note'>Baseado em score de saude e status de drift.</small></article>"
            + "<article class='exec-card insights-card'><span class='exec-kicker'>Insights</span><ul class='exec-list'>"
            + highlights_html
            + "</ul></article>"
            + "<article class='exec-card actions-card'><span class='exec-kicker'>Proximos passos</span><div class='action-box'><ul class='action-list'>"
            + actions_html
            + "</ul></div><small class='exec-note'><span class='info-tip'><span class='info-icon'>i</span><span class='tip-bubble'>PSI acima de 0.25 sugere mudanca relevante de distribuicao.</span></span> Priorizar acao preventiva.</small></article>"
            + "</div></section>"
        )

    def assemble_dashboard(self, ctx: dict[str, Any], sections: list[str], payload: dict[str, Any]) -> str:
        """Monta o HTML final do dashboard com secoes, estilos e scripts."""
        status = self._drift_status(ctx["drift"])
        fairness_link = "<a href='#fairness'>Equidade</a>" if payload.get("fairness", {}).get("data") else ""
        theme_bootstrap = "<script>(function(){try{if(localStorage.getItem('dashboardTheme')==='light'){document.documentElement.classList.add('pref-theme-light');}}catch(e){}})();</script>"
        return "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>Dashboard de Monitoramento</title>" + theme_bootstrap + self._styles() + "</head><body><aside id='sidebar' class='sidebar'><div class='sidebar-head'><div class='sidebar-title'>Monitoramento do Modelo</div><div class='sidebar-subtitle'>Passos Mágicos</div></div><nav class='sidebar-nav'><a href='#risk-segmentation'>Impacto Operacional</a><a href='#model-performance'>Performance do Modelo</a><a href='#drift-analysis'>Drift dos Dados</a>" + fairness_link + "<a href='#explainability'>Explicabilidade</a><a href='#governance'>Governanca</a><a href='#conclusion'>Conclusao e Recomendacao</a></nav><div class='sidebar-divider'></div><div class='sidebar-actions'><div class='theme-row'><span class='theme-row-label'>Tema</span><label class='theme-switch' for='theme-toggle'><input id='theme-toggle' type='checkbox' aria-label='Alternar tema claro e escuro'/><span class='theme-slider'></span></label></div></div></aside><div class='content-shell'><header class='topbar'><div><h1>Dashboard Profissional de Monitoramento</h1><p class='topbar-subtitle'>Visao consolidada de saude do modelo, risco operacional e governanca.</p></div><span class='status-badge " + status["css"] + "'>" + status["label"] + "</span></header><main class='container'>" + "".join(sections) + "</main></div><!-- plotly.js inline -->" + self._plotly_runtime() + "<script>window.dashboardPayload = " + json.dumps(payload, ensure_ascii=True) + ";</script>" + self._plot_script() + self._ui_enhancement_script() + "</body></html>"

    def _build_payload(self, ctx: dict[str, Any]) -> dict[str, Any]:
        t, temp, d, rows = ctx["train"], ctx["temporal"], ctx["drift"], ctx["rows"]
        cm = t.get("confusion_matrix")
        if isinstance(cm, dict):
            cmz = [[int(cm.get("tn", 0) or 0), int(cm.get("fp", 0) or 0)], [int(cm.get("fn", 0) or 0), int(cm.get("tp", 0) or 0)]]
        else:
            folds = temp.get("folds", []) if isinstance(temp.get("folds"), list) else []
            cmz = folds[-1].get("confusion_matrix") if folds else None
            if not (isinstance(cmz, list) and len(cmz) == 2):
                cmz = None
        roc = t.get("roc_curve", {}) if isinstance(t.get("roc_curve"), dict) else {}
        pr = t.get("pr_curve", {}) if isinstance(t.get("pr_curve"), dict) else {}
        dist = t.get("probability_distribution", {}) if isinstance(t.get("probability_distribution"), dict) else {}
        bins, pos, neg = dist.get("bins", []), dist.get("positive", []), dist.get("negative", [])
        centers = []
        if isinstance(bins, list) and len(bins) > 1:
            for i in range(len(bins) - 1):
                left, right = self._f(bins[i]), self._f(bins[i + 1])
                centers.append(round((left + right) / 2.0, 4) if left is not None and right is not None else str(i))
        trade = t.get("threshold_tradeoff_curve", []) if isinstance(t.get("threshold_tradeoff_curve"), list) else []
        tt = [self._f(x.get("threshold")) for x in trade if isinstance(x, dict) and self._f(x.get("threshold")) is not None]
        tf1 = [self._f(x.get("f1")) for x in trade if isinstance(x, dict) and self._f(x.get("threshold")) is not None]
        trc = [self._f(x.get("recall")) for x in trade if isinstance(x, dict) and self._f(x.get("threshold")) is not None]
        tpr = [self._f(x.get("precision")) for x in trade if isinstance(x, dict) and self._f(x.get("threshold")) is not None]
        psi_items = []
        pbf = d.get("psi_by_feature", {})
        if isinstance(pbf, dict) and pbf:
            for feat, payload in pbf.items():
                psi = self._f(payload.get("psi")) if isinstance(payload, dict) else None
                if psi is not None:
                    psi_items.append((str(feat), psi))
        else:
            for row in d.get("psi_metrics", []) if isinstance(d.get("psi_metrics"), list) else []:
                if isinstance(row, dict):
                    psi = self._f(row.get("psi"))
                    if psi is not None:
                        psi_items.append((str(row.get("feature", "n/d")), psi))
        psi_items.sort(key=lambda x: x[1], reverse=True)
        psi_colors = ["#22c55e" if psi < 0.1 else "#f59e0b" if psi <= 0.25 else "#ef4444" for _, psi in psi_items]
        top_traces = []
        for feat, payload in sorted([(str(k), v) for k, v in pbf.items() if isinstance(v, dict)], key=lambda x: self._f(x[1].get("psi")) or 0.0, reverse=True)[:5]:
            bb, rr, cc = payload.get("bins", []), payload.get("reference_distribution", []), payload.get("current_distribution", [])
            if isinstance(bb, list) and len(bb) == len(rr) + 1 == len(cc) + 1:
                x = []
                for i in range(len(bb) - 1):
                    left, right = self._f(bb[i]), self._f(bb[i + 1])
                    x.append(round((left + right) / 2.0, 4) if left is not None and right is not None else str(i))
                top_traces += [{"type": "scatter", "mode": "lines", "x": x, "y": rr, "name": f"{feat} ref"}, {"type": "scatter", "mode": "lines", "x": x, "y": cc, "name": f"{feat} atual"}]
        dh = d.get("drift_history", []) if isinstance(d.get("drift_history"), list) else []
        dxt, dyy = [str(r.get("timestamp")) for r in dh if isinstance(r, dict) and r.get("timestamp") is not None and self._f(r.get("avg_psi")) is not None], [self._f(r.get("avg_psi")) for r in dh if isinstance(r, dict) and r.get("timestamp") is not None and self._f(r.get("avg_psi")) is not None]
        hr = d.get("high_risk_rate_history", []) if isinstance(d.get("high_risk_rate_history"), list) else []
        hxt, hyy = [str(r.get("timestamp")) for r in hr if isinstance(r, dict) and r.get("timestamp") is not None and self._f(r.get("rate")) is not None], [self._f(r.get("rate")) for r in hr if isinstance(r, dict) and r.get("timestamp") is not None and self._f(r.get("rate")) is not None]
        strategic = d.get("strategic_monitoring", {}) if isinstance(d.get("strategic_monitoring"), dict) else {}
        cur_rate = self._f(strategic.get("current_high_risk_rate_pct"))
        s = [str(r.get("strategy", "n/d")) for r in rows]
        return {
            "rocCurve": {"data": ([{"type": "scatter", "mode": "lines", "x": roc.get("fpr", []), "y": roc.get("tpr", []), "name": "ROC", "line": {"color": "#3b82f6", "width": 3}}, {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "name": "Aleatorio", "line": {"color": "#9ca3af", "dash": "dash"}}] if isinstance(roc.get("fpr"), list) and isinstance(roc.get("tpr"), list) and roc.get("fpr") else []) or ([{"type": "scatter", "mode": "lines+markers", "x": [0, 0.2, 1], "y": [0, max(0.0, min(1.0, 2 * (self._f(t.get("auc")) or 0.5) - 0.05)), 1], "name": "ROC (aproximada)", "line": {"color": "#3b82f6", "width": 3}}] if self._f(t.get("auc")) is not None else []), "layout": self._dark_layout("Curva ROC")},
            "prCurve": {"data": ([{"type": "scatter", "mode": "lines", "x": pr.get("recall", []), "y": pr.get("precision", []), "name": "Precisao-Recall", "line": {"color": "#22c55e", "width": 3}}] if isinstance(pr.get("precision"), list) and isinstance(pr.get("recall"), list) and pr.get("precision") else []), "layout": self._dark_layout("Curva Precisao-Recall")},
            "confusionMatrix": {"data": ([{"type": "heatmap", "z": cmz, "x": ["Previsto 0", "Previsto 1"], "y": ["Real 0", "Real 1"], "colorscale": "Viridis"}] if cmz else []), "layout": self._dark_layout("Matriz de Confusao")},
            "probabilityDistribution": {"data": ([{"type": "bar", "x": centers, "y": pos, "name": "Positiva", "marker": {"color": "#22c55e"}}, {"type": "bar", "x": centers, "y": neg, "name": "Negativa", "marker": {"color": "#9ca3af"}}] if centers and isinstance(pos, list) and isinstance(neg, list) else []), "layout": {**self._dark_layout("Distribuicao de probabilidades"), "barmode": "group"}},
            "thresholdTradeoff": {"data": ([{"type": "scatter", "mode": "lines", "x": tt, "y": tf1, "name": "F1", "line": {"color": "#3b82f6", "width": 3}}, {"type": "scatter", "mode": "lines", "x": tt, "y": trc, "name": "Recall", "line": {"color": "#a855f7", "width": 3}}, {"type": "scatter", "mode": "lines", "x": tt, "y": tpr, "name": "Precisao", "line": {"color": "#f59e0b", "width": 3}}] if tt else []), "layout": self._dark_layout("Limiar vs F1/Recall")},
            "psiBar": {"data": ([{"type": "bar", "x": [x[0] for x in psi_items], "y": [x[1] for x in psi_items], "name": "PSI", "marker": {"color": psi_colors}}] if psi_items else []), "layout": self._dark_layout("PSI por Feature (ordenado)")},
            "topDriftDistribution": {"data": top_traces, "layout": self._dark_layout("Comparacao de distribuicao top 5 features")},
            "avgPsiEvolution": {"data": ([{"type": "scatter", "mode": "lines+markers", "x": dxt, "y": dyy, "name": "avg_psi", "line": {"color": "#f59e0b", "width": 3}}] if dxt else []), "layout": self._dark_layout("Evolucao temporal de avg_psi")},
            "highRiskEvolution": {"data": ([{"type": "scatter", "mode": "lines+markers", "x": hxt, "y": hyy, "name": "ALTO_RISCO", "line": {"color": "#ef4444", "width": 3}}] if hxt else []), "layout": self._dark_layout("Evolucao da taxa ALTO_RISCO")},
            "riskSegmentation": {"data": ([{"type": "pie", "labels": ["ALTO_RISCO", "OUTROS"], "values": [max(0.0, min(100.0, cur_rate)), 100.0 - max(0.0, min(100.0, cur_rate))], "hole": 0.5, "marker": {"colors": ["#ef4444", "#6b7280"]}}] if cur_rate is not None else []), "layout": self._dark_layout("Distribuicao atual por segmento")},
            "thresholdComparison": {"data": ([{"type": "bar", "x": s, "y": [self._f(r.get("recall")) or 0.0 for r in rows], "name": "Recall", "marker": {"color": ["#22c55e" if x == ctx["active"] else "#3b82f6" for x in s]}}, {"type": "bar", "x": s, "y": [self._f(r.get("f1_score")) or 0.0 for r in rows], "name": "F1", "marker": {"color": "#a855f7"}}, {"type": "scatter", "mode": "lines+markers", "x": s, "y": [self._f(r.get("threshold")) or 0.0 for r in rows], "name": "Limiar", "line": {"color": "#f59e0b", "width": 3}, "yaxis": "y2"}] if rows else []), "layout": {**self._dark_layout("Comparacao visual entre estrategias"), "barmode": "group", "yaxis2": {"overlaying": "y", "side": "right", "gridcolor": "#1f2430"}}},
            "featureImportance": self._feature_importance_payload(ctx["train"]),
            "fairness": self._fairness_payload(ctx["train"]),
        }

    def _feature_importance_payload(self, train: dict[str, Any]) -> dict[str, Any]:
        parsed = []
        ranking = train.get("feature_importance_ranking")
        if isinstance(ranking, list):
            for it in ranking:
                if isinstance(it, dict) and it.get("feature") is not None and self._f(it.get("importance")) is not None:
                    parsed.append((str(it["feature"]), self._f(it["importance"])))
        if not parsed and isinstance(train.get("feature_importance"), dict):
            for feat, val in train["feature_importance"].items():
                if self._f(val) is not None:
                    parsed.append((str(feat), self._f(val)))
        if not parsed:
            return {"data": [], "layout": {"title": "Importancia global das variaveis"}}
        parsed = sorted(parsed, key=lambda x: x[1], reverse=True)[:10]
        feature_names = [x[0] for x in parsed]
        importance_values = [x[1] for x in parsed]
        chart_height = max(420, 160 + (len(feature_names) * 34))

        def _pretty_label(value: str) -> str:
            raw = value.strip()
            raw = re.sub(r"(?i)^(num|cat|flg|flag|vlr|pct|qtd|tmp|dt|cod)_+", "", raw)
            raw = re.sub(r"(?i)_+(num|cat|flg|flag|vlr|pct|qtd|tmp|dt|cod)$", "", raw)
            snake_spaced = raw.replace("_", " ").replace("-", " ").strip()
            humanized = re.sub(r"(?<!^)(?=[A-Z])", " ", snake_spaced)
            words = []
            for token in humanized.split():
                low = token.lower()
                if low in {"id", "cpf", "cep", "uf", "ddd"}:
                    words.append(low.upper())
                else:
                    words.append(low.capitalize())
            label = " ".join(words)
            return label if len(label) <= 32 else f"{label[:29]}..."

        display_labels = [_pretty_label(name) for name in feature_names]
        max_label_len = max((len(label) for label in display_labels), default=0)
        left_margin = min(240, max(120, int(max_label_len * 5.2)))
        marker_colors = ["#334155"] * len(display_labels)
        if marker_colors:
            marker_colors[0] = "#a855f7"
        if len(marker_colors) > 1:
            marker_colors[1] = "#8b5cf6"
        if len(marker_colors) > 2:
            marker_colors[2] = "#7c3aed"
        return {
            "data": [
                {
                    "type": "bar",
                    "orientation": "h",
                    "x": importance_values,
                    "y": display_labels,
                    "customdata": feature_names,
                    "name": "Importancia",
                    "marker": {"color": marker_colors},
                    "hovertemplate": "<b>%{customdata}</b><br>Importancia: %{x:.4f}<extra></extra>",
                }
            ],
            "layout": {
                **self._dark_layout("Importancia global das variaveis"),
                "height": chart_height,
                "margin": {"l": left_margin, "r": 32, "t": 58, "b": 46},
                "showlegend": False,
                "hoverlabel": {"bgcolor": "#111827", "bordercolor": "#374151", "font": {"color": "#f3f4f6", "size": 12}},
                "xaxis": {"title": "Importancia relativa", "gridcolor": "#202633", "zeroline": False, "tickformat": ".3f", "automargin": True},
                "yaxis": {
                    "autorange": "reversed",
                    "automargin": True,
                    "tickfont": {"size": 12, "color": "#e5e7eb"},
                    "ticklabeloverflow": "allow",
                    "ticklabelstandoff": 4,
                },
                "bargap": 0.38,
            },
        }

    def _fairness_payload(self, train: dict[str, Any]) -> dict[str, Any]:
        gm = train.get("group_metrics", {})
        if not isinstance(gm, dict) or not gm:
            return {"data": [], "layout": self._dark_layout("Equidade por grupo")}
        selected_name, selected = None, None
        for k, v in gm.items():
            if isinstance(v, dict) and v:
                selected_name, selected = str(k), v
                break
        if selected is None:
            return {"data": [], "layout": self._dark_layout("Equidade por grupo")}
        rows = []
        for group, metric in selected.items():
            if isinstance(metric, dict):
                label = str(group).replace("_", " ").strip().title()
                rows.append(
                    (
                        label,
                        self._f(metric.get("recall")) or 0.0,
                        self._f(metric.get("precision")) or 0.0,
                        self._f(metric.get("f1_score")) or 0.0,
                    )
                )
        if not rows:
            return {"data": [], "layout": self._dark_layout("Equidade por grupo")}
        rows.sort(key=lambda x: x[3], reverse=True)
        groups = [r[0] for r in rows]
        recall_vals = [r[1] for r in rows]
        precision_vals = [r[2] for r in rows]
        f1_vals = [r[3] for r in rows]
        max_group_len = max((len(group) for group in groups), default=0)
        fairness_left_margin = min(160, max(72, int(max_group_len * 3.6)))
        all_scores = recall_vals + precision_vals + f1_vals
        max_score = max(all_scores) if all_scores else 1.0
        x_upper = min(1.0, max(0.15, max_score * 1.08))
        chart_height = max(420, 170 + (len(groups) * 36))
        return {
            "data": [
                {"type": "bar", "orientation": "h", "y": groups, "x": recall_vals, "name": "Recall", "marker": {"color": "#10b981"}},
                {"type": "bar", "orientation": "h", "y": groups, "x": precision_vals, "name": "Precisao", "marker": {"color": "#38bdf8"}},
                {"type": "bar", "orientation": "h", "y": groups, "x": f1_vals, "name": "F1", "marker": {"color": "#a855f7"}},
            ],
            "layout": {
                **self._dark_layout(f"Equidade por grupo ({selected_name})"),
                "height": chart_height,
                "barmode": "group",
                "margin": {"l": fairness_left_margin, "r": 16, "t": 58, "b": 46},
                "xaxis": {"title": "Score", "range": [0, x_upper], "tickformat": ".0%", "gridcolor": "#202633", "automargin": True},
                "yaxis": {"autorange": "reversed", "automargin": True, "tickfont": {"size": 12}, "ticklabeloverflow": "allow", "ticklabelstandoff": 0},
                "hoverlabel": {"bgcolor": "#111827", "bordercolor": "#374151", "font": {"color": "#f3f4f6", "size": 12}},
                "bargap": 0.18,
                "bargroupgap": 0.06,
            },
        }

    def _threshold_table(self, rows: list[dict[str, Any]], active: str) -> str:
        if not rows:
            return "<div class='empty'>Sem comparativo persistido de estrategias.</div>"
        body = []
        for r in rows:
            strategy = str(r.get("strategy", "n/d"))
            css = "tag-ok" if strategy == active else "tag-warning"
            body.append("<tr><td><span class='tag " + css + "'>" + html.escape(strategy) + "</span></td><td>" + self._pct(r.get("recall")) + "</td><td>" + self._pct(r.get("f1_score")) + "</td><td>" + self._num(r.get("threshold"), 4) + "</td><td>" + self._num(r.get("high_risk_estimated"), 0) + "</td><td>" + self._num(r.get("human_reviews"), 0) + "</td></tr>")
        return "<div class='table-wrap'><table id='threshold-table'><thead><tr><th>Estrategia</th><th>Recall</th><th>F1</th><th>Limiar</th><th>Alto Risco Est.</th><th>Revisoes Humanas</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    def _drift_table(self, drift: dict[str, Any]) -> str:
        rows = []
        pbf = drift.get("psi_by_feature", {})
        if isinstance(pbf, dict) and pbf:
            for feat, payload in pbf.items():
                if isinstance(payload, dict) and self._f(payload.get("psi")) is not None:
                    rows.append((str(feat), self._f(payload.get("psi"))))
        else:
            pm = drift.get("psi_metrics", [])
            if isinstance(pm, list):
                for row in pm:
                    if isinstance(row, dict) and self._f(row.get("psi")) is not None:
                        rows.append((str(row.get("feature", "n/d")), self._f(row.get("psi"))))
        if not rows:
            return "<div class='empty'>Sem dados de drift estruturado.</div>"
        rows.sort(key=lambda x: x[1], reverse=True)
        body = []
        for feat, psi in rows:
            cls = "tag-danger" if psi > 0.25 else "tag-warning" if psi >= 0.1 else "tag-ok"
            txt = "Critico" if psi > 0.25 else "Atencao" if psi >= 0.1 else "Estavel"
            body.append(f"<tr><td>{html.escape(feat)}</td><td>{psi:.4f}</td><td><span class='tag {cls}'>{txt}</span></td></tr>")
        return "<div class='table-wrap'><table><thead><tr><th>Feature</th><th>PSI</th><th>Status</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    def _model_health(self, ctx: dict[str, Any]) -> dict[str, Any]:
        t = ctx.get("train", {})
        temporal = ctx.get("temporal", {})
        drift = ctx.get("drift", {})
        f1 = (self._f(t.get("f1_score")) or 0.0) * 100.0
        recall = (self._f(t.get("recall")) or 0.0) * 100.0
        folds = temporal.get("folds", []) if isinstance(temporal.get("folds"), list) else []
        f1_series = [self._f(f.get("f1")) for f in folds if isinstance(f, dict) and self._f(f.get("f1")) is not None]
        if len(f1_series) > 1:
            mean = sum(f1_series) / len(f1_series)
            variance = sum((x - mean) ** 2 for x in f1_series) / len(f1_series)
            std = variance ** 0.5
            temporal_stability = max(0.0, min(100.0, 100.0 - std * 250.0))
        else:
            temporal_stability = 70.0
        drift_values = []
        pbf = drift.get("psi_by_feature", {})
        if isinstance(pbf, dict) and pbf:
            drift_values = [self._f(v.get("psi")) for v in pbf.values() if isinstance(v, dict) and self._f(v.get("psi")) is not None]
        if not drift_values:
            pm = drift.get("psi_metrics", [])
            drift_values = [self._f(v.get("psi")) for v in pm if isinstance(v, dict) and self._f(v.get("psi")) is not None] if isinstance(pm, list) else []
        avg_psi = sum(drift_values) / len(drift_values) if drift_values else 0.1
        drift_stability = max(0.0, min(100.0, 100.0 - (avg_psi * 220.0)))
        score = max(0.0, min(100.0, (f1 * 0.35) + (recall * 0.25) + (temporal_stability * 0.2) + (drift_stability * 0.2)))
        label = "Excelente" if score >= 85 else "Estavel" if score >= 70 else "Atencao" if score >= 50 else "Critico"
        return {"score": round(score, 1), "label": label, "avg_psi": round(avg_psi, 4)}

    def _resolve_active_strategy(self, train: dict[str, Any], cmp_data: Any) -> str:
        if isinstance(cmp_data, dict):
            for key in ("active_strategy", "selected_strategy", "strategy"):
                candidate = cmp_data.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        strategy = train.get("threshold_strategy")
        return strategy.strip() if isinstance(strategy, str) and strategy.strip() else "n/d"

    def _resolve_threshold_rows(self, train: dict[str, Any], cmp_data: Any, active: str) -> list[dict[str, Any]]:
        rows = []
        if isinstance(cmp_data, dict):
            for key in ("strategies", "comparisons", "rows", "data"):
                value = cmp_data.get(key)
                if isinstance(value, list):
                    rows = [self._normalize_threshold_row(item) for item in value]
                    break
            if not rows and cmp_data.get("strategy"):
                rows = [self._normalize_threshold_row(cmp_data)]
        elif isinstance(cmp_data, list):
            rows = [self._normalize_threshold_row(item) for item in cmp_data]
        rows = [row for row in rows if row]
        return rows or [{"strategy": active, "recall": self._f(train.get("recall")), "f1_score": self._f(train.get("f1_score")), "threshold": self._f(train.get("risk_threshold")), "high_risk_estimated": None, "human_reviews": None}]

    def _normalize_threshold_row(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        strategy = raw.get("strategy") or raw.get("estrategia") or raw.get("name")
        recall = raw.get("recall") if raw.get("recall") is not None else raw.get("recall_pct")
        f1 = raw.get("f1_score")
        if f1 is None:
            f1 = raw.get("f1")
        if f1 is None:
            f1 = raw.get("f1_pct")
        return {"strategy": str(strategy) if strategy is not None else "n/d", "recall": self._ratio(recall), "f1_score": self._ratio(f1), "threshold": self._f(raw.get("threshold")), "high_risk_estimated": self._f(raw.get("high_risk_estimated") if raw.get("high_risk_estimated") is not None else raw.get("alto_risco_estimado")), "human_reviews": self._f(raw.get("human_reviews") if raw.get("human_reviews") is not None else raw.get("revisoes_humanas"))}

    def _drift_status(self, drift: dict[str, Any]) -> dict[str, str]:
        status = drift.get("drift_status")
        if isinstance(status, str):
            n = status.strip().lower()
            if n == "critico":
                return {"label": "Status de Drift: Critico", "label_short": "Critico", "css": "status-danger"}
            if n == "moderado":
                return {"label": "Status de Drift: Moderado", "label_short": "Moderado", "css": "status-warning"}
        strategic = drift.get("strategic_monitoring", {})
        if bool(strategic.get("significant_shift_alert", False)):
            return {"label": "Status de Drift: Critico", "label_short": "Critico", "css": "status-danger"}
        if drift.get("psi_alerts"):
            return {"label": "Status de Drift: Moderado", "label_short": "Moderado", "css": "status-warning"}
        return {"label": "Status de Drift: Estavel", "label_short": "Estavel", "css": "status-ok"}

    @classmethod
    def _plotly_runtime(cls) -> str:
        if cls._PLOTLY_JS_CACHE is None:
            cls._PLOTLY_JS_CACHE = "window.Plotly={newPlot:function(){return null;}};" if get_plotlyjs is None else get_plotlyjs()
        return f"<script>{cls._PLOTLY_JS_CACHE}</script>"

    def _plot_script(self) -> str:
        base = {"paper_bgcolor": "#101522", "plot_bgcolor": "#101522", "font": {"color": "#e8ecf3", "family": "'Segoe UI Variable','Segoe UI',Inter,Roboto,system-ui,sans-serif"}, "margin": {"l": 56, "r": 24, "t": 52, "b": 56}, "xaxis": {"gridcolor": "#2a3245", "zerolinecolor": "#2a3245"}, "yaxis": {"gridcolor": "#2a3245", "zerolinecolor": "#2a3245"}, "legend": {"orientation": "h", "y": 1.12, "bgcolor": "#101522", "bordercolor": "#2a3245", "borderwidth": 1}}
        return (
            "<script>(function(){"
            "const p=window.dashboardPayload||(typeof dashboardPayload!=='undefined'?dashboardPayload:{});"
            "const b=" + json.dumps(base, ensure_ascii=True) + ";"
            "const hasPlotly=!!window.Plotly;"
            "const chartIds=['roc-curve-plot','pr-curve-plot','confusion-matrix-heatmap','probability-distribution-plot','threshold-tradeoff-plot','psi-feature-bar','drift-distribution-top5','avg-psi-evolution','high-risk-evolution','risk-segmentation-chart','threshold-strategy-compare','feature-importance-global','fairness-group-chart'];"
            "function palette(light){return light?{paper:'#ffffff',plot:'#ffffff',font:'#1f2937',grid:'#d5deea',legend:'#ffffff',legendBorder:'#cfd9e8'}:{paper:'#101522',plot:'#101522',font:'#e8ecf3',grid:'#2a3245',legend:'#101522',legendBorder:'#2a3245'};}"
            "function r(i,d,l){const e=document.getElementById(i);if(!e){return;}if(!hasPlotly){return;}if(!d||!d.length){e.innerHTML='<div class=\"empty\">Sem dados suficientes para este grafico.</div>';return;}Plotly.newPlot(i,d,Object.assign({},b,l||{}),{responsive:true,displayModeBar:false});}"
            "function applyPlotTheme(light){if(!hasPlotly){return;}const c=palette(light);for(const id of chartIds){const e=document.getElementById(id);if(!e||!e.data||!e.data.length){continue;}Plotly.relayout(id,{paper_bgcolor:c.paper,plot_bgcolor:c.plot,font:{color:c.font,family:\"'Segoe UI Variable','Segoe UI',Inter,Roboto,system-ui,sans-serif\"},'xaxis.gridcolor':c.grid,'xaxis.zerolinecolor':c.grid,'yaxis.gridcolor':c.grid,'yaxis.zerolinecolor':c.grid,'legend.bgcolor':c.legend,'legend.bordercolor':c.legendBorder});}}"
            "function setTheme(light,persist){document.body.classList.toggle('theme-light',light);document.documentElement.classList.toggle('pref-theme-light',light);const sw=document.getElementById('theme-toggle');if(sw){sw.checked=light;}applyPlotTheme(light);if(persist){try{localStorage.setItem('dashboardTheme',light?'light':'dark');}catch(e){}}}"
            "r('roc-curve-plot',p.rocCurve?.data,p.rocCurve?.layout);r('pr-curve-plot',p.prCurve?.data,p.prCurve?.layout);r('confusion-matrix-heatmap',p.confusionMatrix?.data,p.confusionMatrix?.layout);r('probability-distribution-plot',p.probabilityDistribution?.data,p.probabilityDistribution?.layout);r('threshold-tradeoff-plot',p.thresholdTradeoff?.data,p.thresholdTradeoff?.layout);r('psi-feature-bar',p.psiBar?.data,p.psiBar?.layout);r('drift-distribution-top5',p.topDriftDistribution?.data,p.topDriftDistribution?.layout);r('avg-psi-evolution',p.avgPsiEvolution?.data,p.avgPsiEvolution?.layout);r('high-risk-evolution',p.highRiskEvolution?.data,p.highRiskEvolution?.layout);r('risk-segmentation-chart',p.riskSegmentation?.data,p.riskSegmentation?.layout);r('threshold-strategy-compare',p.thresholdComparison?.data,p.thresholdComparison?.layout);r('feature-importance-global',p.featureImportance?.data,p.featureImportance?.layout);r('fairness-group-chart',p.fairness?.data,p.fairness?.layout);"
            "const preferredLight=(document.documentElement.classList.contains('pref-theme-light'));setTheme(preferredLight,false);"
            "const toggle=document.getElementById('theme-toggle');if(toggle){toggle.addEventListener('change',function(){setTheme(!!toggle.checked,true);});}"
            "})();</script>"
        )

    @staticmethod
    def _ui_enhancement_script() -> str:
        return (
            "<script>(function(){"
            "function toPartsSP(v){if(!v){return null;}const d=new Date(v);if(Number.isNaN(d.getTime())){return null;}const fmt=new Intl.DateTimeFormat('pt-BR',{timeZone:'America/Sao_Paulo',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});const parts=fmt.formatToParts(d);const out={};for(const p of parts){out[p.type]=p.value;}return out;}"
            "function formatDateBRSP(v){const p=toPartsSP(v);if(!p){return 'n/d';}return `${p.day}/${p.month}/${p.year} ${p.hour}:${p.minute}`;}"
            "function formatVersionFromDate(v){const p=toPartsSP(v);if(!p){return 'v----.--.--:--:--:--';}return `v${p.year}.${p.month}.${p.day}:${p.hour}:${p.minute}:${p.second}`;}"
            "function initGovernance(){const root=document.querySelector('[data-governance]');if(!root){return;}const generatedRaw=root.getAttribute('data-generated-at');const versionRaw=root.getAttribute('data-version-source')||generatedRaw;const dateEl=root.querySelector('[data-governance-generated]');const versionEl=root.querySelector('[data-governance-version]');if(dateEl){dateEl.textContent=formatDateBRSP(generatedRaw);}if(versionEl){versionEl.textContent=formatVersionFromDate(versionRaw);}}"
            "function initConclusion(){const root=document.getElementById('conclusion');if(!root){return;}const score=Number(root.getAttribute('data-health-score'));const drift=String(root.getAttribute('data-drift-status')||'').toLowerCase();const badge=root.querySelector('[data-verdict-badge]');const reason=root.querySelector('[data-verdict-reason]');if(!badge||!reason){return;}badge.classList.remove('verdict-approved','verdict-attention','verdict-rejected');if(drift.includes('crit')||(!Number.isNaN(score)&&score<60)){badge.textContent='Reprovado';badge.classList.add('verdict-rejected');reason.textContent='Risco alto de degradacao. Exigir plano corretivo imediato.';return;}if(drift.includes('moder')||(!Number.isNaN(score)&&score<75)){badge.textContent='Atencao';badge.classList.add('verdict-attention');reason.textContent='Operacao aceitavel, com pontos de vigilancia ativa.';return;}badge.textContent='Aprovado';badge.classList.add('verdict-approved');reason.textContent='Modelo estavel, com sinais consistentes de confiabilidade.';}"
            "initGovernance();initConclusion();"
            "})();</script>"
        )

    def _dark_layout(self, title: str) -> dict[str, Any]:
        return {"title": title, "paper_bgcolor": "#101522", "plot_bgcolor": "#101522", "font": {"color": "#e8ecf3", "family": "'Segoe UI Variable','Segoe UI',Inter,Roboto,system-ui,sans-serif"}, "margin": {"l": 56, "r": 24, "t": 52, "b": 56}, "legend": {"orientation": "h", "y": 1.12, "bgcolor": "#101522", "bordercolor": "#2a3245", "borderwidth": 1}, "xaxis": {"gridcolor": "#2a3245", "zerolinecolor": "#2a3245"}, "yaxis": {"gridcolor": "#2a3245", "zerolinecolor": "#2a3245"}}

    def _section_title(self, title: str, tooltip: str | None = None) -> str:
        return "<div class='section-header'><h2>" + html.escape(title) + (" " + self._info_icon(tooltip) if tooltip else "") + "</h2></div>"

    def _info_icon(self, text: str | None) -> str:
        if not text:
            return ""
        escaped = html.escape(text)
        return "<span class='info-tip' title='" + escaped + "'><span class='info-icon'>ⓘ</span><span class='tip-bubble'>" + escaped + "</span></span>"

    @staticmethod
    def _styles() -> str:
        return """<style>
        :root{
          --bg:#05070d;
          --bg_alt:#080a10;
          --surface:#101522;
          --surface_alt:#151d2e;
          --surface_soft:#1a2438;
          --accent:#4f7cff;
          --accent_soft:#2f4e9e;
          --success:#22c55e;
          --warning:#f59e0b;
          --danger:#ef4444;
          --text_primary:#e9eef7;
          --text_secondary:#a7b1c2;
          --text_muted:#8793a8;
          --border:#253048;
          --shadow:0 14px 34px rgba(2,6,16,.36);
          --radius_lg:18px;
          --radius_md:12px;
        }
        body.theme-light,
        html.pref-theme-light{
          --bg:#f4f7fc;
          --bg_alt:#e8eef8;
          --surface:#ffffff;
          --surface_alt:#f7faff;
          --surface_soft:#eef3fb;
          --accent:#315dcb;
          --accent_soft:#24469c;
          --text_primary:#182234;
          --text_secondary:#4b5b74;
          --text_muted:#62738f;
          --border:#d2dceb;
          --shadow:0 12px 26px rgba(26,39,62,.10);
        }
        *{box-sizing:border-box}
        html,body{margin:0;padding:0;min-height:100%}
        body{
          background:
            radial-gradient(1200px 700px at 92% -10%, color-mix(in srgb, var(--accent) 12%, transparent), transparent 60%),
            linear-gradient(165deg,var(--bg) 0%,var(--bg_alt) 55%,var(--bg) 100%);
          color:var(--text_primary);
          font-family:'Segoe UI Variable','Segoe UI',Inter,Roboto,system-ui,sans-serif;
          line-height:1.5;
        }
        :is(.section,.metric-card,.plot-card,.hint-card,.table-wrap,.theme-row,.sidebar-head,.sidebar-nav a,.status-badge,.info-icon){
          border-radius:var(--radius_md);
        }
        :is(.metric-card,.plot-card,.hint-card,.sidebar-nav a,.info-icon,.theme-row,.status-badge,.table-wrap,.section){
          transition:all .2s ease;
        }
        .sidebar{
          position:fixed;left:0;top:0;bottom:0;width:272px;padding:22px 16px 18px;
          background:color-mix(in srgb, var(--bg) 88%, #000 12%);
          border-right:1px solid var(--border);
          backdrop-filter:blur(10px);
          display:flex;flex-direction:column;gap:14px;
        }
        .sidebar-head{
          padding:14px 14px 15px;
          border:1px solid var(--border);
          background:linear-gradient(180deg,var(--surface_alt),var(--surface_soft));
        }
        .sidebar-title{
          font-size:12px;font-weight:760;letter-spacing:.1em;text-transform:uppercase;
          color:var(--text_primary);margin-bottom:6px;
        }
        .sidebar-subtitle{
          color:var(--text_muted);font-size:12px;letter-spacing:.02em;
          font-weight:600;line-height:1.45;
        }
        .sidebar-nav{display:grid;gap:8px}
        .sidebar-nav a{
          color:var(--text_secondary);text-decoration:none;padding:11px 13px;
          font-size:12px;font-weight:680;letter-spacing:.015em;border:1px solid transparent;
          position:relative;
        }
        .sidebar-nav a::before{
          content:'';
          position:absolute;
          left:0;top:20%;
          width:3px;height:60%;
          border-radius:999px;
          background:transparent;
          transition:background .2s ease;
        }
        .sidebar-nav a:hover{
          background:color-mix(in srgb, var(--accent) 16%, transparent);
          border-color:color-mix(in srgb, var(--accent) 32%, transparent);
          color:var(--text_primary);
          transform:translateX(3px);
        }
        .sidebar-nav a:hover::before{
          background:color-mix(in srgb, var(--accent) 70%, transparent);
        }
        .sidebar-divider{
          border-top:1px solid var(--border);
          margin:4px 8px;
        }
        .sidebar-actions{margin-top:auto;padding:0 8px 2px}
        .theme-row{
          display:flex;align-items:center;justify-content:space-between;gap:12px;
          border:1px solid var(--border);padding:11px 13px;
          background:linear-gradient(180deg,var(--surface_alt),var(--surface_soft));
        }
        .theme-row-label{
          font-size:12px;font-weight:700;color:var(--text_primary);letter-spacing:.03em;
          text-transform:uppercase;
        }
        .theme-switch{
          position:relative;display:inline-flex;align-items:center;cursor:pointer;
          width:52px;height:30px;
        }
        .theme-switch input{
          position:absolute;opacity:0;pointer-events:none;
        }
        .theme-slider{
          position:absolute;inset:0;border-radius:999px;
          background:linear-gradient(180deg,var(--surface_soft),var(--surface_alt));
          border:1px solid var(--border);
          transition:all .2s ease;
          box-shadow:inset 0 0 0 1px rgba(0,0,0,.08);
        }
        .theme-slider::before{
          content:'';
          position:absolute;left:3px;top:3px;width:22px;height:22px;border-radius:50%;
          background:linear-gradient(180deg,var(--surface),var(--surface_alt));
          box-shadow:0 2px 6px rgba(0,0,0,.35);
          border:1px solid var(--border);
          transition:transform .2s ease;
        }
        .theme-slider::after{
          content:'';
          position:absolute;right:9px;top:9px;width:6px;height:6px;border-radius:50%;
          background:var(--text_muted);
          box-shadow:0 0 10px color-mix(in srgb, var(--text_muted) 55%, transparent);
          opacity:.35;transition:all .22s ease;
        }
        .theme-switch input:checked + .theme-slider{
          background:linear-gradient(180deg,color-mix(in srgb, var(--accent) 46%, var(--surface_alt)),color-mix(in srgb, var(--accent_soft) 52%, var(--surface_alt)));
          border-color:color-mix(in srgb, var(--accent) 35%, var(--border));
        }
        .theme-switch input:checked + .theme-slider::before{
          transform:translateX(22px);
          background:linear-gradient(180deg,var(--surface),color-mix(in srgb, var(--surface_alt) 86%, var(--accent)));
        }
        .theme-switch input:checked + .theme-slider::after{
          left:10px;right:auto;opacity:1;
          background:color-mix(in srgb, var(--accent) 58%, var(--surface));
          box-shadow:0 0 12px color-mix(in srgb, var(--accent) 55%, transparent);
        }
        .content-shell{margin-left:272px;min-height:100vh}
        .topbar{
          display:flex;justify-content:space-between;align-items:flex-start;gap:20px;
          padding:36px 34px 12px;
        }
        .topbar h1{
          margin:0;font-size:32px;line-height:1.1;letter-spacing:-.015em;font-weight:760;
          color:var(--text_primary);
        }
        .topbar-subtitle{
          margin:10px 0 0;
          color:var(--text_secondary);
          font-size:14px;
          line-height:1.62;
          max-width:740px;
        }
        .status-badge{
          padding:9px 14px;border-radius:999px;font-size:12px;font-weight:760;border:1px solid;
          background:color-mix(in srgb, var(--surface) 88%, transparent);
          letter-spacing:.04em;
          text-transform:uppercase;
          white-space:nowrap;
        }
        .status-ok{color:var(--success);border-color:rgba(34,197,94,.45)}
        .status-warning{color:var(--warning);border-color:rgba(245,158,11,.45)}
        .status-danger{color:var(--danger);border-color:rgba(239,68,68,.45)}
        .container{max-width:1400px;margin:0 auto;padding:0 30px 52px}
        .section{
          background:linear-gradient(175deg,var(--surface),var(--surface_alt));
          border:1px solid var(--border);
          border-radius:var(--radius_lg);
          padding:30px;
          margin-top:22px;
          box-shadow:var(--shadow);
        }
        .section:hover{
          transform:translateY(-1px);
          border-color:color-mix(in srgb, var(--accent) 24%, var(--border));
        }
        .section-header{
          display:flex;
          align-items:flex-start;
          justify-content:space-between;
          gap:10px;
        }
        .section-header h2{
          display:flex;align-items:center;gap:10px;margin:0;
          font-size:27px;line-height:1.18;font-weight:760;letter-spacing:-.012em;
        }
        .section-intro{
          color:var(--text_secondary);font-size:14px;line-height:1.72;margin:14px 0 0;
          max-width:1020px;
        }
        .status-header{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap}
        .model-name{margin:0;font-size:34px;line-height:1.08;text-align:center;letter-spacing:-.01em}
        .env-badge{
          display:inline-flex;align-items:center;justify-content:center;padding:5px 10px;border-radius:999px;
          font-size:11px;font-weight:700;letter-spacing:.05em;border:1px solid var(--border)
        }
        .env-prod{background:rgba(34,197,94,.14);color:#86efac;border-color:rgba(34,197,94,.35)}
        .env-dev{background:rgba(245,158,11,.14);color:#fcd34d;border-color:rgba(245,158,11,.35)}
        .status-updated{text-align:center;margin:8px 0 0;color:var(--text_muted);font-size:14px}
        .score-copy{font-size:24px;font-weight:720;line-height:1.2}
        .status-diagnosis-line{margin:8px auto 0;max-width:860px;text-align:center;color:var(--text_secondary);font-size:14px;line-height:1.6}
        .metrics-grid{
          display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
          gap:16px;margin-top:18px;
        }
        .metric-card{
          background:linear-gradient(180deg,var(--surface_alt),var(--surface_soft));
          border:1px solid var(--border);padding:16px 16px 14px;
        }
        .metric-card:hover{
          border-color:color-mix(in srgb, var(--accent) 30%, var(--border));
          transform:translateY(-2px);
        }
        .metric-label{
          font-size:10px;color:color-mix(in srgb, var(--text_muted) 86%, transparent);text-transform:uppercase;letter-spacing:.12em;
          font-weight:760;
        }
        .metric-value{font-size:30px;font-weight:760;margin-top:8px;line-height:1.12}
        .metric-small{font-size:16px;word-break:break-word}
        .plot-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px}
        .plot-stack{display:grid;gap:11px}
        .plot-card{
          min-height:340px;border:1px solid var(--border);
          background:linear-gradient(180deg,var(--surface),color-mix(in srgb, var(--surface_alt) 72%, var(--surface)));
          margin-top:12px;overflow:hidden;
          width:100%;
          position:relative;
        }
        .plot-card:hover{
          border-color:color-mix(in srgb, var(--accent) 30%, var(--border));
        }
        .explainability-plot{min-height:460px}
        .fairness-plot{min-height:420px}
        .hint-card{
          margin-top:8px;padding:12px 14px;border:1px solid var(--border);
          background:linear-gradient(180deg,color-mix(in srgb, var(--surface_soft) 86%, transparent),color-mix(in srgb, var(--surface_alt) 78%, transparent));
          color:var(--text_secondary);font-size:13px;line-height:1.72;
        }
        .hint-card:hover{
          border-color:color-mix(in srgb, var(--accent) 24%, var(--border));
        }
        .table-wrap{
          overflow:auto;border:1px solid var(--border);margin-top:18px;
          background:color-mix(in srgb, var(--surface) 88%, transparent);
        }
        table{width:100%;border-collapse:separate;border-spacing:0;min-width:720px}
        th,td{padding:14px 14px;text-align:left;border-bottom:1px solid color-mix(in srgb, var(--border) 95%, transparent);font-size:13px;line-height:1.45}
        th{
          color:var(--text_muted);font-size:10px;text-transform:uppercase;letter-spacing:.12em;
          font-weight:700;background:color-mix(in srgb, var(--surface_alt) 90%, transparent);
          position:sticky;top:0;
        }
        tbody tr:nth-child(even){background:color-mix(in srgb, var(--surface_soft) 58%, transparent)}
        tbody tr:hover{
          background:color-mix(in srgb, var(--accent) 13%, transparent);
          transform:scale(1.001);
        }
        .tag{padding:4px 8px;border-radius:999px;font-size:10px;font-weight:760;letter-spacing:.08em;text-transform:uppercase}
        .tag-ok{color:var(--success)}
        .tag-warning{color:var(--warning)}
        .tag-danger{color:var(--danger)}
        .empty{padding:16px;color:var(--text_secondary)}
        .conclusion-list{margin:14px 0 0;padding-left:20px;display:grid;gap:10px}
        .info-tip{position:relative;display:inline-flex;align-items:center}
        .info-icon{
          display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:999px;
          border:1px solid var(--border);color:var(--text_secondary);font-size:11px;cursor:help;
          background:color-mix(in srgb, var(--surface_alt) 85%, transparent);
          box-shadow:0 3px 10px color-mix(in srgb, var(--bg) 22%, transparent);
        }
        .info-tip:hover .info-icon{
          color:var(--text_primary);
          border-color:color-mix(in srgb, var(--accent) 28%, var(--border));
        }
        .tip-bubble{
          position:absolute;left:22px;top:50%;transform:translateY(-50%);
          min-width:280px;max-width:380px;padding:11px 12px;border-radius:10px;
          border:1px solid var(--border);background:linear-gradient(180deg,var(--surface_alt),var(--surface_soft));color:var(--text_secondary);
          font-size:12px;line-height:1.58;opacity:0;pointer-events:none;transition:opacity .2s ease-in, transform .2s ease;
          box-shadow:var(--shadow);
        }
        .info-tip:hover .tip-bubble{opacity:1;transform:translateY(-50%) translateX(2px)}
        .audit-section .section-intro{margin-bottom:16px}
        .audit-bar{
          display:grid;
          grid-template-columns:repeat(3,minmax(0,1fr));
          gap:14px;
        }
        .audit-item{
          display:flex;
          align-items:flex-start;
          gap:12px;
          padding:14px;
          border:1px solid var(--border);
          background:linear-gradient(180deg,var(--surface_alt),var(--surface_soft));
        }
        .audit-item:hover{
          border-color:color-mix(in srgb, var(--accent) 30%, var(--border));
          transform:translateY(-2px);
        }
        .audit-icon{
          width:28px;
          height:28px;
          min-width:28px;
          display:inline-flex;
          align-items:center;
          justify-content:center;
          border:1px solid color-mix(in srgb, var(--accent) 24%, var(--border));
          background:color-mix(in srgb, var(--surface) 88%, transparent);
          color:color-mix(in srgb, var(--accent) 76%, var(--text_secondary));
          font-size:12px;
          font-weight:760;
        }
        .audit-icon svg{width:16px;height:16px}
        .audit-meta{display:grid;gap:6px}
        .audit-label{
          font-size:10px;
          text-transform:uppercase;
          letter-spacing:.12em;
          color:var(--text_muted);
          font-weight:760;
        }
        .audit-value{
          font-size:15px;
          line-height:1.3;
          color:var(--text_primary);
          font-weight:680;
          word-break:break-word;
        }
        .technical{
          font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;
          letter-spacing:.02em;
        }
        .executive-summary .section-intro{margin-bottom:16px}
        .exec-grid{
          display:grid;
          grid-template-columns:1.15fr 1fr 1.15fr;
          gap:16px;
        }
        .exec-card{
          border:1px solid var(--border);
          padding:18px;
          background:linear-gradient(180deg,var(--surface_alt),var(--surface_soft));
        }
        .exec-kicker{
          font-size:10px;
          text-transform:uppercase;
          letter-spacing:.12em;
          color:var(--text_muted);
          font-weight:760;
        }
        .verdict-badge{
          margin-top:10px;
          display:inline-flex;
          padding:9px 14px;
          border-radius:999px;
          border:1px solid var(--border);
          font-weight:760;
          text-transform:uppercase;
          letter-spacing:.08em;
        }
        .verdict-approved{color:var(--success);border-color:color-mix(in srgb, var(--success) 42%, var(--border))}
        .verdict-attention{color:var(--warning);border-color:color-mix(in srgb, var(--warning) 42%, var(--border))}
        .verdict-rejected{color:var(--danger);border-color:color-mix(in srgb, var(--danger) 42%, var(--border))}
        .verdict-reason{margin:10px 0 0;color:var(--text_secondary);line-height:1.6}
        .exec-list,.action-list{
          margin:10px 0 0;
          padding-left:0;
          list-style:none;
          display:grid;
          gap:9px;
        }
        .exec-list li,.action-list li{
          position:relative;
          padding-left:18px;
          color:var(--text_secondary);
          line-height:1.58;
        }
        .exec-list li::before,.action-list li::before{
          content:'';
          position:absolute;
          left:0;
          top:.54em;
          width:8px;
          height:8px;
          border-radius:999px;
          background:color-mix(in srgb, var(--accent) 75%, transparent);
        }
        .action-box{
          margin-top:10px;
          padding:12px;
          border:1px solid var(--border);
          background:color-mix(in srgb, var(--surface_soft) 90%, transparent);
        }
        .exec-note{
          margin-top:10px;
          display:block;
          color:var(--text_muted);
          line-height:1.5;
        }
        @media (max-width:1100px){
          .sidebar{position:static;width:auto;padding:14px 10px}
          .content-shell{margin-left:0}
          .plot-grid{grid-template-columns:1fr}
          .topbar{padding:22px 18px 8px}
          .topbar h1{font-size:28px}
          .container{padding:0 14px 28px}
          .section{padding:22px}
          .model-name{font-size:28px}
          .tip-bubble{left:0;top:24px;transform:none;max-width:90vw}
          .audit-bar{grid-template-columns:repeat(2,minmax(0,1fr))}
          .exec-grid{grid-template-columns:1fr}
        }
        @media (max-width:760px){
          .metrics-grid{grid-template-columns:1fr}
          .metric-value{font-size:27px}
          .section-header h2{font-size:23px}
          .status-badge{font-size:11px}
          .audit-bar{grid-template-columns:1fr}
        }
        </style>"""

    @staticmethod
    def _f(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _ratio(self, value: Any) -> float | None:
        number = self._f(value)
        if number is None:
            return None
        return number / 100.0 if number > 1.0 else number

    def _pct(self, value: Any) -> str:
        number = self._f(value)
        return "n/d" if number is None else f"{number * 100:.1f}%"

    def _num(self, value: Any, decimals: int) -> str:
        number = self._f(value)
        return "n/d" if number is None else f"{number:.{decimals}f}"

    def _read_json(self, filename: str) -> Any:
        path = os.path.join(self.monitoring_dir, filename)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            logger.warning(f"Falha ao ler {filename}: {exc}")
            return {}

    @staticmethod
    def _write_text(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
