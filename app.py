"""Application Streamlit : analyse statistique des matchs (modèle v4).   Lancer en local : streamlit run app.py"""
import datetime as dt
import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from footpred import preparer, config as C
from footpred.markets import cote_juste
from footpred.decision import classement, trois_decisions, PALIERS
from footpred import statistiques as ST

st.set_page_config(page_title="Analyse statistique football", page_icon="⚽", layout="wide")
BLEU, GRIS, ORANGE = "#2a78d6", "#8a8984", "#eb6834"
FAMILLES = ["Résultat", "Double chance", "Buts", "Corners", "Cartons"]


@st.cache_resource(ttl=dt.timedelta(hours=12), show_spinner="Chargement des données, calcul des ratings et entraînement des modèles (≈ 30 s)…")
def charger():
    # données fraîches à chaque rechargement (toutes les 12 h) ; FOOTPRED_SOURCE permet d'utiliser un fichier local
    return preparer(os.environ.get("FOOTPRED_SOURCE", C.DATA_URL), cache=None)


@st.cache_data(ttl=dt.timedelta(hours=12))
def equipes_par_ligue(_date_max):
    X = charger().d.X
    out = {}
    for lg in C.LIGUES:
        x = X[X.Division == lg]; s = x.Season.max()
        eq = set(x[x.Season == s].HomeTeam) | set(x[x.Season == s].AwayTeam)
        if len(eq) < 16:                                          # début de saison : compléter avec la saison précédente
            eq |= set(x[x.Season == s - 1].HomeTeam)
        out[lg] = sorted(eq)
    return out


def lire_csv(chemin, **kw):
    return pd.read_csv(chemin, **kw) if os.path.exists(chemin) else None


pred = charger()
EQ = equipes_par_ligue(str(pred.d.date_max))
TOUTES = sorted({e for v in EQ.values() for e in v})
PAL = lire_csv("results/tables/decisions_2021_2026_paliers.csv", index_col=0)

# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    st.title("⚽ Analyse statistique")
    page = st.radio("Navigation", ["Statistiques des équipes", "Analyse d'un match", "Analyse d'une journée", "Fiabilité du modèle", "Méthode"],
                    label_visibility="collapsed")
    retard = (dt.date.today() - pred.d.date_max.date()).days
    st.caption(f"Données au **{pred.d.date_max.date()}**" + (f" · ⚠ {retard} jours de retard" if retard > 7 else ""))
    if st.button("Recharger les données"):
        st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    st.divider()
    st.caption("Projet académique. Les probabilités décrivent l'incertitude d'un match ; elles ne sont pas des conseils de pari. "
               "Sur 2021-2026, le modèle reste en moyenne moins précis que les cotes et aucune règle de sélection n'a été rentable.")


def choix_match(cle, defaut_ligue=1):
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    lg = c1.selectbox("Championnat", list(C.LIGUES), format_func=C.LIGUES.get, index=defaut_ligue, key=f"lg_{cle}")
    aff = {"SP1": ("Barcelona", "Real Madrid"), "E0": ("Arsenal", "Liverpool"), "I1": ("Inter", "Milan"),
           "D1": ("Bayern Munich", "Dortmund"), "F1": ("Paris SG", "Marseille")}.get(lg, ("", ""))
    dom = c2.selectbox("Équipe à domicile", EQ[lg], index=EQ[lg].index(aff[0]) if aff[0] in EQ[lg] else 0, key=f"dom_{cle}")
    autres = [e for e in EQ[lg] if e != dom]
    ext = c3.selectbox("Équipe à l'extérieur", autres, index=autres.index(aff[1]) if aff[1] in autres else 0, key=f"ext_{cle}")
    return lg, dom, ext


def barre_1x2(r, dom, ext):
    df = pd.DataFrame({"issue": [f"1 · {dom}", "X · nul", f"2 · {ext}"], "p": [r["1"], r["X"], r["2"]], "ordre": [0, 1, 2]})
    base = alt.Chart(df).encode(y=alt.Y("issue:N", sort=alt.SortField("ordre"), title=None),
                                x=alt.X("p:Q", axis=alt.Axis(format="%", tickCount=5), title=None, scale=alt.Scale(domain=[0, 1])),
                                tooltip=[alt.Tooltip("issue:N"), alt.Tooltip("p:Q", format=".1%")])
    return (base.mark_bar(cornerRadiusEnd=4, height=26, color=BLEU) + base.mark_text(align="left", dx=6).encode(text=alt.Text("p:Q", format=".1%"))).properties(height=130)


def carte_scores(M, dom, ext, n=6):
    d = pd.DataFrame([(i, j, M[i, j]) for i in range(n) for j in range(n)], columns=[dom, ext, "p"])
    base = alt.Chart(d).encode(x=alt.X(f"{ext}:O", title=f"Buts {ext}", axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{dom}:O", title=f"Buts {dom}"),
                               tooltip=[alt.Tooltip(f"{dom}:O"), alt.Tooltip(f"{ext}:O"), alt.Tooltip("p:Q", format=".1%")])
    heat = base.mark_rect(stroke="white", strokeWidth=2).encode(color=alt.Color("p:Q", scale=alt.Scale(scheme="blues"), legend=None))
    txt = base.mark_text(fontSize=11).encode(text=alt.Text("p:Q", format=".1%"), color=alt.condition("datum.p > 0.06", alt.value("white"), alt.value("#0b0b0b")))
    return (heat + txt).properties(height=300)


def legende_paliers():
    if PAL is None:
        return
    txt = " · ".join(f"**{i}** {PAL.loc[i, 'taux_réalisé']:.0%}" for i in PAL.index)
    n = f"{int(PAL['décisions'].sum()):,}".replace(",", " ")
    st.caption(f"Fiabilité historique des paliers (2021-2026, {n} décisions testées) : taux de réalisation observé — {txt}")


PCT = lambda lab=None: st.column_config.NumberColumn(lab, format="%.1f %%")

# ================================================================== 1. statistiques descriptives
if page == "Statistiques des équipes":
    st.header("Statistiques des équipes")
    st.caption("Statistiques descriptives calculées sur les derniers matchs joués : aucune prédiction, uniquement ce qui s'est passé.")
    lg, dom, ext = choix_match("stats")
    n = st.select_slider("Nombre de derniers matchs pris en compte", options=[5, 10, 15, 20, 38], value=10)
    T = ST.comparaison(pred.d.D, dom, ext, n)
    pct = [i for i in T.index if i.startswith("%") or i in ("victoires", "nuls", "défaites")]
    aff = T.copy().astype(object)
    for i in T.index:
        aff.loc[i] = [("" if pd.isna(v) else f"{v:.0%}") if i in pct else (v if isinstance(v, str) else ("" if pd.isna(v) else (f"{int(v)}" if i == "matchs" else f"{v:.2f}")))
                      for v in T.loc[i]]
    st.dataframe(aff.rename_axis("indicateur").reset_index(), hide_index=True, width="stretch", height=820,
                 column_config={"indicateur": st.column_config.TextColumn(width="medium")})

    st.subheader("Corners et cartons par match")
    sel = ["Corners obtenus (pour)", "Corners obtenus (contre)", "Cartons jaunes (pour)", "Cartons jaunes (contre)"]
    G = T.loc[sel, [f"{dom} (tous)", f"{ext} (tous)", "moyenne de la ligue"]].reset_index().melt("index", var_name="série", value_name="moyenne")
    ch = alt.Chart(G).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("série:N", title=None), x=alt.X("moyenne:Q", title=None),
        color=alt.Color("série:N", scale=alt.Scale(range=[BLEU, ORANGE, GRIS]), legend=alt.Legend(orient="top", title=None)),
        row=alt.Row("index:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=["série", "index", alt.Tooltip("moyenne:Q", format=".2f")]).properties(height=70)
    st.altair_chart(ch, width="stretch")

    st.subheader(f"Confrontations directes {dom} – {ext}")
    H = ST.confrontations(pred.d.D, dom, ext, 10)
    if H.empty:
        st.info("Aucune confrontation dans les données (depuis 2005).")
    else:
        st.dataframe(H, hide_index=True, width="stretch")
    with st.expander(f"Détail des {n} derniers matchs"):
        for e in (dom, ext):
            M = ST.matchs_equipe(pred.d.D, e, n)
            st.markdown(f"**{e}**")
            st.dataframe(M.assign(date=M.date.dt.date)[["date", "lieu", "adversaire", "résultat", "Buts marqués_pour", "Buts marqués_contre",
                                                         "Corners obtenus_pour", "Corners obtenus_contre", "Cartons jaunes_pour", "Cartons jaunes_contre"]]
                         .iloc[::-1], hide_index=True, width="stretch")

# ================================================================== 2. analyse d'un match
elif page == "Analyse d'un match":
    st.header("Analyse d'un match")
    lg, dom, ext = choix_match("match")
    with st.expander("Ajuster l'Elo (blessures, rotation…)"):
        a, b = pred.d.etat(dom), pred.d.etat(ext)
        o1, o2 = st.columns(2)
        elo_d = o1.number_input(f"Elo {dom}", value=float(round(a["elo"])), step=10.0)
        elo_e = o2.number_input(f"Elo {ext}", value=float(round(b["elo"])), step=10.0)
    r = pred.predire(dom, ext, elo_dom=elo_d, elo_ext=elo_e)
    M = r.pop("_M")

    onglets = st.tabs(["Probabilités", "Corners et cartons", "Cotes et décisions"])
    with onglets[0]:
        m1, m2, m3, m4 = st.columns(4)
        for col, lab, k in ((m1, f"Victoire {dom}", "1"), (m2, "Match nul", "X"), (m3, f"Victoire {ext}", "2")):
            col.metric(lab, f"{r[k]:.1%}"); col.caption(f"cote juste {cote_juste(r[k]):.2f}")
        m4.metric("Buts attendus", f"{r['buts_dom']:.2f} – {r['buts_ext']:.2f}"); m4.caption(f"total {r['buts_dom'] + r['buts_ext']:.2f}")
        st.altair_chart(barre_1x2(r, dom, ext), width="stretch")
        st.info(f"**Fiabilité {r['fiabilite']}** — quand le modèle est aussi confiant, son issue favorite s'est réalisée "
                f"**{r['precision_historique']:.0%}** du temps sur 2021-2026.")
        if r["fin_de_saison"]:
            st.warning("Fin de saison : ces matchs sont historiquement moins prévisibles (enjeux, rotations).")
        g1, g2 = st.columns([1, 1.1])
        with g1:
            lignes = [("Plus de 1,5 but", r["over15"]), ("Plus de 2,5 buts", r["over25"]), ("Plus de 3,5 buts", r["over35"]),
                      ("Les deux équipes marquent", r["btts"]), ("But en 1re mi-temps", r["but_1re_MT"]),
                      (f"Cage inviolée {dom}", r["cs_dom"]), (f"Cage inviolée {ext}", r["cs_ext"])]
            st.dataframe(pd.DataFrame({"Marché": [l[0] for l in lignes], "Probabilité": [100 * l[1] for l in lignes], "Cote juste": [cote_juste(l[1]) for l in lignes]}),
                         hide_index=True, width="stretch",
                         column_config={"Probabilité": st.column_config.ProgressColumn(format="%.1f %%", min_value=0, max_value=100),
                                        "Cote juste": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Scores les plus probables : " + " · ".join(f"**{s}** {q:.1%}" for q, s in r["scores"]))
        with g2:
            st.altair_chart(carte_scores(M, dom, ext), width="stretch")

    with onglets[1]:
        c, j, rg = r["corners"], r["jaunes"], r["rouges"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Corners attendus", f"{c['total']:.1f}"); k1.caption(f"{dom} {c['dom']:.1f} · {ext} {c['ext']:.1f}")
        k2.metric("Cartons jaunes attendus", f"{j['total']:.1f}"); k2.caption(f"{dom} {j['dom']:.1f} · {ext} {j['ext']:.1f}")
        k3.metric("Au moins un carton rouge", f"{rg['au_moins_un']:.0%}"); k3.caption(f"cote juste {cote_juste(rg['au_moins_un']):.2f}")
        loi = pd.DataFrame({"corners": np.arange(len(c["loi"])), "p": c["loi"]}).query("corners <= 20")
        st.altair_chart(alt.Chart(loi).mark_bar(color=BLEU, cornerRadiusEnd=3).encode(
            x=alt.X("corners:O", title="Nombre total de corners", axis=alt.Axis(labelAngle=0)), y=alt.Y("p:Q", title=None, axis=alt.Axis(format="%")),
            tooltip=["corners", alt.Tooltip("p:Q", format=".1%")]).properties(height=200, title="Loi du nombre total de corners"), width="stretch")
        def tableau(dico, titre):
            return pd.DataFrame({"Marché": [f"{titre} plus de {s_:g}".replace(".", ",") for s_ in dico], "Probabilité": [100 * q for q in dico.values()],
                                 "Cote juste": [cote_juste(q) for q in dico.values()]})
        cfg = {"Probabilité": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100), "Cote juste": st.column_config.NumberColumn(format="%.2f")}
        t1, t2 = st.columns(2)
        t1.dataframe(pd.concat([tableau(c["plus_de"], "Corners :"), tableau(c["dom_plus_de"], f"{dom} :"), tableau(c["ext_plus_de"], f"{ext} :")]),
                     hide_index=True, width="stretch", column_config=cfg)
        t2.dataframe(pd.concat([tableau(j["plus_de"], "Jaunes :"), tableau(j["dom_plus_de"], f"{dom} :"), tableau(j["ext_plus_de"], f"{ext} :")]),
                     hide_index=True, width="stretch", column_config=cfg)
        st.caption("Incertitude élevée : corrélation prévu/réel sur 2021-2026 de 0,14 pour les corners, 0,28 pour les jaunes et 0,13 pour les rouges "
                   "(0,21 pour le total de buts). L'arbitre, absent des données, pèse beaucoup sur les cartons.")

    with onglets[2]:
        st.markdown("Tous les marchés du match, **du moins risqué au plus risqué** (risque = probabilité que l'issue ne se réalise pas). "
                    "Saisis les cotes d'un bookmaker dans la dernière colonne pour comparer : la *valeur attendue* indique le gain moyen "
                    "pour une mise de 1 **si les probabilités du modèle étaient exactes**.")
        fam = st.multiselect("Familles de marchés", FAMILLES, default=FAMILLES)
        base = classement(r)
        base = base[base.famille.isin(fam)]
        cle = f"cotes_{dom}_{ext}"
        saisie = st.data_editor(base[["famille", "marché", "probabilité", "risque", "cote juste", "cote bookmaker"]].assign(probabilité=100 * base["probabilité"]),
                                key=cle, hide_index=True, width="stretch", height=420,
                                disabled=["famille", "marché", "probabilité", "risque", "cote juste"],
                                column_config={"probabilité": PCT("probabilité"), "cote juste": st.column_config.NumberColumn(format="%.2f"),
                                               "cote bookmaker": st.column_config.NumberColumn("cote bookmaker ✏️", min_value=1.01, max_value=100.0, format="%.2f")})
        legende_paliers()
        cotes = {m: c_ for m, c_ in zip(saisie["marché"], saisie["cote bookmaker"]) if pd.notna(c_)}
        T = classement(r, cotes); T = T[T.famille.isin(fam)]
        st.subheader("Trois décisions statistiques")
        D3 = trois_decisions(T)
        cols = st.columns(3)
        for col, (_, d) in zip(cols, D3.iterrows()):
            with col.container(border=True):
                st.caption(f"Décision {d['profil']}")
                st.markdown(f"**{d['marché']}**")
                st.markdown(f"### {d['probabilité']:.0%}")
                st.caption(f"risque {d['risque'][4:]} · cote juste {d['cote juste']:.2f}"
                           + (f" · valeur attendue {d['valeur attendue']:+.1%}" if pd.notna(d["valeur attendue"]) else ""))
        if cotes:
            V = T.dropna(subset=["cote bookmaker"]).sort_values("probabilité", ascending=False)
            st.subheader("Comparaison avec les cotes saisies")
            st.dataframe(V[["marché", "probabilité", "risque", "cote juste", "cote bookmaker", "prob. implicite", "écart (pts)", "valeur attendue"]]
                         .assign(**{"probabilité": 100 * V["probabilité"], "prob. implicite": 100 * V["prob. implicite"], "valeur attendue": 100 * V["valeur attendue"]}),
                         hide_index=True, width="stretch",
                         column_config={"probabilité": PCT(), "prob. implicite": PCT("prob. implicite (avec marge)"),
                                        "cote juste": st.column_config.NumberColumn(format="%.2f"), "cote bookmaker": st.column_config.NumberColumn(format="%.2f"),
                                        "écart (pts)": st.column_config.NumberColumn(format="%+.1f"), "valeur attendue": st.column_config.NumberColumn(format="%+.1f %%")})
            st.caption("Une valeur attendue positive signifie seulement que le modèle est plus optimiste que le bookmaker. Sur 2021-2026, ces écarts "
                       "ont surtout reflété les erreurs du modèle : sélectionner les « valeurs positives » a perdu de l'argent en moyenne (page Fiabilité).")

# ================================================================== 3. analyse d'une journée
elif page == "Analyse d'une journée":
    st.header("Analyse d'une journée")
    st.caption("Saisis les affiches. Pour chaque match, l'application donne trois décisions statistiques, puis classe les matchs "
               "du moins risqué au plus risqué selon la décision prudente.")
    if "journee" not in st.session_state:
        st.session_state.journee = pd.DataFrame({"Domicile": ["Arsenal", "Barcelona", "Paris SG", "Inter"], "Extérieur": ["Chelsea", "Sevilla", "Lyon", "Milan"]})
    saisie = st.data_editor(st.session_state.journee, num_rows="dynamic", width="stretch",
                            column_config={"Domicile": st.column_config.SelectboxColumn(options=TOUTES, required=True),
                                           "Extérieur": st.column_config.SelectboxColumn(options=TOUTES, required=True)})
    fam = st.multiselect("Familles de marchés autorisées pour les décisions", FAMILLES, default=["Résultat", "Double chance", "Buts", "Corners"])
    lignes, detail = [], []
    for _, m in saisie.dropna().iterrows():
        if m.Domicile == m.Extérieur:
            continue
        try:
            r = pred.predire(m.Domicile, m.Extérieur); r.pop("_M")
        except ValueError as e:
            st.error(str(e)); continue
        T = classement(r); D3 = trois_decisions(T, fam).set_index("profil")
        lignes.append({"Match": f"{m.Domicile} – {m.Extérieur}", "1": r["1"], "X": r["X"], "2": r["2"],
                       "Décision prudente": D3.loc["prudente", "marché"], "p prudente": D3.loc["prudente", "probabilité"], "risque": D3.loc["prudente", "risque"],
                       "Décision équilibrée": D3.loc["équilibrée", "marché"], "p équilibrée": D3.loc["équilibrée", "probabilité"],
                       "Décision audacieuse": D3.loc["audacieuse", "marché"], "p audacieuse": D3.loc["audacieuse", "probabilité"],
                       "Buts attendus": r["buts_dom"] + r["buts_ext"], "Corners attendus": r["corners"]["total"], "Jaunes attendus": r["jaunes"]["total"]})
        detail.append(T.assign(match=f"{m.Domicile} – {m.Extérieur}"))
    if lignes:
        J = pd.DataFrame(lignes).sort_values("p prudente", ascending=False).reset_index(drop=True)
        J.index = J.index + 1; J.index.name = "rang"
        pcols = ["1", "X", "2", "p prudente", "p équilibrée", "p audacieuse"]
        st.subheader("Matchs classés du moins risqué au plus risqué")
        st.dataframe(J.assign(**{c: 100 * J[c] for c in pcols}), width="stretch",
                     column_config={**{c: PCT() for c in pcols}, **{c: st.column_config.NumberColumn(format="%.1f") for c in ["Buts attendus", "Corners attendus", "Jaunes attendus"]}})
        legende_paliers()
        st.caption("Profil prudent = issue la plus probable (hors issues quasi certaines > 95 %) ; équilibré ≈ 60 % ; audacieux ≈ 40 %. "
                   "Une décision peu risquée a une cote faible : le risque et le gain potentiel varient toujours en sens inverse.")
        sortie = pd.concat(detail)
        sortie.insert(0, "horodatage", dt.datetime.now().strftime("%Y-%m-%d %H:%M")); sortie.insert(1, "donnees_au", str(pred.d.date_max.date()))
        st.download_button("Télécharger l'analyse complète (CSV)", sortie.to_csv(index=False).encode("utf-8"), file_name=f"analyse_journee_{dt.date.today()}.csv", mime="text/csv")

# ================================================================== 4. fiabilité
elif page == "Fiabilité du modèle":
    st.header("Fiabilité du modèle — test 2021/22 → 2025/26")
    st.caption("Chaque saison est prédite par un modèle entraîné uniquement sur les saisons précédentes (walk-forward).")
    if PAL is not None:
        st.subheader("Les paliers de risque tiennent-ils leurs promesses ?")
        st.dataframe(PAL.assign(**{c: 100 * PAL[c] for c in ["probabilité_annoncée", "taux_réalisé", "rendement_aux_cotes"]}), width="stretch",
                     column_config={"probabilité_annoncée": PCT("probabilité annoncée"), "taux_réalisé": PCT("taux réalisé"),
                                    "rendement_aux_cotes": st.column_config.NumberColumn("rendement aux cotes réelles", format="%+.1f %%")})
        st.caption("Sur environ 265 000 décisions (30 marchés × 8 822 matchs), la probabilité annoncée correspond au taux réellement observé dans chaque palier : "
                   "le classement par risque est fiable. Mais là où les cotes réelles existent (1X2, plus/moins de 2,5 buts), le rendement est négatif à "
                   "tous les paliers : la marge du bookmaker et la qualité du marché l'emportent.")
    T = pd.read_csv("results/tables/validation_2021_2026_1X2.csv", index_col=0)
    st.subheader("Résultat 1X2")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Log loss v4", f"{T.loc['v4', 'log loss']:.4f}", f"{T.loc['v4', 'log loss'] - T.loc['Elo seul', 'log loss']:+.4f} vs Elo seul", delta_color="inverse")
    k2.metric("Log loss cotes", f"{T.loc['Cotes', 'log loss']:.4f}")
    k3.metric("Bon résultat v4", f"{T.loc['v4', 'bon résultat']:.1%}")
    k4.metric("Matchs testés", f"{int(T.loc['v4', 'matchs']):,}".replace(",", " "))
    st.dataframe(T.drop(columns="matchs"), width="stretch",
                 column_config={"bon résultat": st.column_config.NumberColumn(format="percent"), "skill vs fréquences": st.column_config.NumberColumn(format="percent")})
    st.image("results/figures/validation_2021_2026.png", width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tests de significativité")
        st.dataframe(pd.read_csv("results/tables/validation_2021_2026_tests.csv", index_col=0).round(2), width="stretch")
    with c2:
        st.subheader("Marchés de buts")
        st.dataframe(pd.read_csv("results/tables/validation_2021_2026_buts.csv", index_col=0).map(lambda v: "—" if pd.isna(v) else f"{v:.4f}"), width="stretch")
    S = lire_csv("results/tables/corners_cartons_2021_2026_stats_match.csv", index_col=0)
    if S is not None:
        st.subheader("Corners et cartons")
        st.dataframe(S.round(3), width="stretch")
        st.caption("Δ < 0 avec un IC entièrement négatif : le modèle fait mieux que la simple moyenne de la ligue. Gain réel mais modeste.")
    st.subheader("Saison 2025/26 rejouée semaine par semaine")
    st.image("results/figures/hebdo_2025.png", width="stretch")
    st.subheader("Le marché est-il battable ? (stratégies simulées, rendement par mise de 1)")
    S = pd.read_csv("results/tables/strategies_2021_2026_strategies.csv", index_col=0)
    st.dataframe(S, width="stretch", column_config={c: st.column_config.NumberColumn(format="percent") for c in ["gagnés", "rendement", "IC bas", "IC haut"]})
    st.caption("Aucune stratégie n'a un intervalle de confiance entièrement positif.")

# ================================================================== 5. méthode
else:
    st.markdown(open("docs/methodologie.md", encoding="utf-8").read())
