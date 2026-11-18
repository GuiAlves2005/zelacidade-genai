"""Testes que rodam sem GPU (pytest -q)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zela.avaliacao import corrigir, placar  # noqa: E402
from zela.base import carregar_fontes, carregar_teste, carregar_trechos  # noqa: E402
from zela.busca import Buscador  # noqa: E402
from zela.esquema import extrair_json, validar  # noqa: E402
from zela.llm import GeradorFixo  # noqa: E402
from zela.pipeline import PipelineZela  # noqa: E402
from zela.privacidade import mascarar  # noqa: E402

TRECHOS = carregar_trechos()
TESTE = carregar_teste()


def test_base_integra():
    fontes = carregar_fontes()
    assert len(TRECHOS) >= 40
    assert all(t.fonte in fontes for t in TRECHOS)
    ids = {t.id for t in TRECHOS}
    for p in TESTE:
        assert set(p["trechos_relevantes"]) <= ids, p["id"]
        assert p["deve_recusar"] == (not p["trechos_relevantes"])
    assert sum(p["deve_recusar"] for p in TESTE) == 10


def test_extrair_json_com_lixo_em_volta():
    txt = 'Claro! ```json\n{"resposta": "x {y}", "categoria": "calcada", "risco": "baixo", "fontes": ["T014"], "recusou": false}\n``` fim'
    obj, erro = validar(txt)
    assert erro is None and obj.fontes == ["T014"]
    assert extrair_json("sem json") is None


def test_esquema_rejeita_categoria_e_fonte_invalidas():
    assert validar('{"resposta":"a","categoria":"xyz","risco":"baixo","fontes":[],"recusou":false}')[0] is None
    assert validar('{"resposta":"a","categoria":"calcada","risco":"baixo","fontes":["doc1"],"recusou":false}')[0] is None


def test_mascarar_dados_pessoais_preserva_telefones_publicos():
    s = mascarar("Meu CPF é 123.456.789-00, e-mail joao@ex.com, cel (11) 98765-4321. Liguei 0800 779 0156")
    assert "123.456" not in s and "joao@" not in s and "98765" not in s
    assert "0800 779 0156" in s


def _oraculo(mensagens):
    """Gerador falso: cita o 1º trecho do contexto e copia o texto dele."""
    user = mensagens[-1]["content"]
    if "CONTEXTO" not in user:
        return json.dumps({"resposta": "Não sei.", "categoria": "fora_do_escopo", "risco": "nao_se_aplica",
                           "fontes": [], "recusou": True})
    linha = user.split("\n")[1]
    tid = linha[1:5]
    texto = linha.split(": ", 1)[1]
    return json.dumps({"resposta": texto, "categoria": "buraco_via", "risco": "medio", "fontes": [tid], "recusou": False},
                      ensure_ascii=False)


def test_pipeline_e_placar_ponta_a_ponta():
    busca = Buscador(TRECHOS, "tfidf")
    linhas = []
    for cfg in ["A", "B"]:
        pipe = PipelineZela(cfg, GeradorFixo(_oraculo), busca, top_k=3, limiar=0.13)
        for p in TESTE:
            e = pipe.responder(p["pergunta"]).para_dict()
            linhas.append(corrigir(e, p))
    tab = placar(linhas)
    assert set(tab.index) == {"A", "B"}
    assert tab.loc["A", "taxa_alucinacao"] == 0  # o oráculo sempre recusa sem contexto
    assert tab.loc["B", "acuracia_respondiveis"] > 0
    assert tab.loc["B", "json_valido"] == 1


def test_recusa_por_limiar_nao_chama_modelo():
    chamado = []
    pipe = PipelineZela("B", GeradorFixo(lambda m: chamado.append(1) or "{}"), Buscador(TRECHOS, "tfidf"), limiar=10.0)
    e = pipe.responder("Qual a capital da França?")
    assert e.recusa_por_limiar and e.saida.recusou and not chamado
