import streamlit as st
import json
import os

st.set_page_config(layout="wide", page_title="Validador Factual - Fase I")

INPUT_FILE = "data/injected/pipeline_input.jsonl"
OUTPUT_FILE = "data/gold_standard/validated_dataset.jsonl"

def load_data():
    if not os.path.exists(INPUT_FILE):
        return []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_gold(record):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    st.title("🔬 Laboratório de Construção do Ground Truth Factual")
    
    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    data = load_data()
    if not data:
        st.error("Nenhum dado encontrado. Execute o script de preparação primeiro.")
        return

    if st.session_state.idx >= len(data):
        st.success("🎉 Todas as sentenças foram validadas!")
        return

    item = data[st.session_state.idx]
    
    st.progress(st.session_state.idx / len(data))
    st.markdown(f"**Anotação {st.session_state.idx + 1} de {len(data)}** | Doc: `{item['doc_id']}`")

    col_doc, col_action = st.columns([1, 1.2])

    # --- COLUNA ESQUERDA: DOCUMENTO FONTE ---
    with col_doc:
        st.markdown("### Documento Fonte")
        st.text_area("Use este texto para encontrar a evidência:", 
                     value=item['document_text'], height=450, disabled=True)

    # --- COLUNA DIREITA: AVALIAÇÃO E RECUPERAÇÃO ---
    with col_action:
        st.markdown("### 1. Evidência (Recuperar Primeiro)")
        st.info(f"**Sentença Original do Resumo:**\n\n{item['original_sentence']}")
        
        st.markdown("*Localize a informação acima no texto da esquerda e cole a sentença exata abaixo.*")
        gold_evidence = st.text_area("Sentença de Evidência (Gold Evidence):", height=100)

        st.markdown("---")
        st.markdown("### 2. Avaliação Factual (Injetar Depois)")
        
        # Renderização condicional baseada na flag "is_modified"
        if item['is_modified']:
            st.error(f"⚠️ **Sentença Injetada (LLM):**\n\n{item['injected_sentence']}")
            st.caption(f"O agente tentou inserir um erro de: **{item['target_error_type'].upper()}**")
            
            is_valid = st.radio("A injeção linguística é plausível?", ["Sim", "Não"], horizontal=True)
            
            final_type = st.selectbox(
                "Qual erro foi REALMENTE inserido?", 
                ["entidade", "predicado", "correferência", "outros", "sem_erro"],
                index=["entidade", "predicado", "correferência"].index(item['target_error_type']) if item['target_error_type'] in ["entidade", "predicado", "correferência"] else 3
            )
            
            final_label = "CONTRADICTED" if is_valid == "Sim" and final_type != "sem_erro" else "SUPPORTED"
            
        else:
            st.success("✅ **Sentença Original (Não Modificada).** O agente não injetou erros nesta sentença.")
            final_type = "nenhum"
            final_label = "SUPPORTED"
            is_valid = "Sim"
            
        st.markdown(f"**Label Factual Final:** `{final_label}`")

        # Botão de Salvar
        if st.button("Salvar e Próximo (Ground Truth)", type="primary"):
            if not gold_evidence.strip():
                st.warning("⚠️ Você precisa colar a evidência ouro antes de prosseguir.")
            else:
                record = {
                    "doc_id": item['doc_id'],
                    "sentence_id": item['sentence_id'],
                    "sentence_evaluated": item['injected_sentence'] if item['is_modified'] else item['original_sentence'],
                    "gold_evidence": gold_evidence.strip(),
                    "is_modified": item['is_modified'],
                    "injection_was_plausible": is_valid == "Sim",
                    "error_type": final_type,
                    "label": final_label
                }
                save_gold(record)
                st.session_state.idx += 1
                st.rerun()

if __name__ == "__main__":
    main()