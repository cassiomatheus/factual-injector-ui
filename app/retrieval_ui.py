import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(layout="wide", page_title="Alinhamento Resumo-Documento")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data/processed/retrieval_ready.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data/gold_standard/aligned_evidence.jsonl"

def load_data():
    if not os.path.exists(INPUT_FILE):
        return []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_annotation(record):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    st.title("⚖️ Fase 1: Mapeamento de Evidências (Resumo $\\rightarrow$ Documento)")
    
    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    data = load_data()
    if not data:
        st.error("Nenhum dado encontrado. Execute o pre_processor.py primeiro.")
        return

    if st.session_state.idx >= len(data):
        st.success("🎉 Todas as sentenças do resumo foram avaliadas!")
        return

    item = data[st.session_state.idx]
    
    st.progress(st.session_state.idx / len(data))
    st.markdown(f"**Anotação {st.session_state.idx + 1} de {len(data)}** | Doc: `{item['doc_id']}`")

    # Exibição da Sentença do Resumo (Bilingue)
    st.markdown("### 📝 Sentença do Resumo (Claim)")
    st.info(f"🇺🇸 **EN:** {item['summary_sentence_en']}\n\n🇧🇷 **PT:** {item['summary_sentence_pt']}")
    
    st.markdown("---")
    st.markdown("### 🔍 Top-5 Evidências Recuperadas no Documento")
    st.markdown("Avalie se as sentenças recuperadas oferecem **Apoio**, **Contradição** ou **Nenhum** em relação à sentença do resumo.")

    # Formulário para as Top-K sentenças
    annotations = []
    with st.form("annotation_form"):
        for i, ev in enumerate(item['retrieved_top_k']):
            st.markdown(f"**Evidência {i+1}** *(Bi-encoder: {ev['bi_encoder_score']:.2f} | Cross-encoder: {ev['cross_encoder_score']:.2f})*")
            st.markdown(f"🇺🇸 {ev['text_en']}")
            st.caption(f"🇧🇷 {ev['text_pt']}")
            
            # Opções de validação para cada evidência
            choice = st.radio(
                f"Relação Factual (Evidência {i+1}):", 
                options=["Apoio (Support)", "Contradição (Contradiction)", "Nenhum/Irrelevante"],
                index=2, # Padrão é Nenhum
                key=f"radio_{i}",
                horizontal=True
            )
            annotations.append({
                "doc_sent_id": ev['doc_sent_id'],
                "text_en": ev['text_en'],
                "human_label": choice
            })
            st.divider()
            
        submitted = st.form_submit_button("Salvar Validações e Próximo", type="primary")
        
        if submitted:
            # Filtra apenas as evidências que o humano marcou como Apoio ou Contradição
            valid_evidences = [ann for ann in annotations if ann['human_label'] != "Nenhum/Irrelevante"]
            
            final_record = {
                "doc_id": item['doc_id'],
                "summary_sentence_id": item['summary_sentence_id'],
                "summary_sentence_en": item['summary_sentence_en'],
                "validated_evidences": valid_evidences
            }
            save_annotation(final_record)
            st.session_state.idx += 1
            st.rerun()

if __name__ == "__main__":
    main()
