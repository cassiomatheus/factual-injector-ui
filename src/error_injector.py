import spacy
import random
import json
import argparse
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

class DatasetPreparer:
    def __init__(self, model_name="qwen3.5:9b"):
        # Carrega o segmentador de sentenças
        self.nlp = spacy.load("en_core_web_sm")
        
        # Conexão com Ollama local (temperatura baixa para focar em precisão)
        self.llm = Ollama(model=model_name, temperature=0.1)
        
        self.prompt = PromptTemplate(
            input_variables=["document", "sentence", "error_type", "instruction"],
            template="""Você é um assistente linguístico focado em avaliação factual.
Seu objetivo é alterar uma ÚNICA sentença para que ela se torne FACTUALMENTE INCORRETA em relação ao documento original, preservando a fluidez e a gramática.

Documento de Referência:
{document}

Sentença Original (Correta):
{sentence}

Tipo de Erro a Injetar: {error_type}
Instrução: {instruction}

Forneça APENAS o novo texto da sentença modificada, sem aspas, sem introduções e sem explicações.
Sentença Modificada:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        self.instructions = {
            "entidade": "Substitua uma entidade (pessoa, organização, local) por outra presente no documento, mas incorreta para este contexto.",
            "predicado": "Altere o verbo principal, inverta a ação (ex: de aprovar para rejeitar), adicione negação ou mude a temporalidade.",
            "correferência": "Mude um pronome ou referente para apontar para o sujeito ou objeto errado."
        }

    def process_document(self, doc_id: str, document_text: str, max_sentences=6):
        """Segmenta o documento, escolhe sentenças e gera o dataset (50% original, 50% erro)"""
        doc = self.nlp(document_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]
        
        # Seleciona algumas sentenças aleatórias para este documento
        selected_sentences = random.sample(sentences, min(max_sentences, len(sentences)))
        
        dataset_items = []
        
        for i, sentence in enumerate(selected_sentences):
            # 50% de chance de ser original ou modificada
            is_modified = random.choice([True, False])
            
            item = {
                "doc_id": doc_id,
                "sentence_id": f"{doc_id}_s{i}",
                "document_text": document_text,
                "original_sentence": sentence,
                "is_modified": is_modified,
                "target_error_type": "nenhum",
                "injected_sentence": None
            }
            
            if is_modified:
                error_type = random.choice(list(self.instructions.keys()))
                print(f"[{doc_id}] Injetando erro de {error_type}...")
                
                try:
                    injected = self.chain.run(
                        document=document_text,
                        sentence=sentence,
                        error_type=error_type,
                        instruction=self.instructions[error_type]
                    ).strip()
                    
                    item["target_error_type"] = error_type
                    item["injected_sentence"] = injected
                except Exception as e:
                    print(f"Erro na injeção: {e}")
                    item["is_modified"] = False # Reverte para original em caso de falha
            
            dataset_items.append(item)
            
        return dataset_items

# --- Execução Local ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara registros de exemplo para validação humana.")
    parser.add_argument("--limit", type=int, default=5, help="Quantidade máxima de registros (padrão: 5).")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit deve ser maior que zero")

    # Documento de exemplo; cada sentença selecionada gera um registro.
    sample_doc = (
        "O Ministério da Saúde publicou o relatório em 2022. "
        "O documento analisa os gastos com hospitais públicos. "
        "O comitê aprovou os resultados finais no mês passado. "
        "A Secretaria de Educação apresentou outro relatório sobre escolas públicas. "
        "A equipe do Ministério da Saúde visitou três hospitais em Brasília. "
        "O comitê encaminhou suas recomendações ao Ministério da Saúde."
    )
    
    preparer = DatasetPreparer()
    items = preparer.process_document(doc_id="doc_001", document_text=sample_doc, max_sentences=args.limit)
    
    output_path = Path(__file__).resolve().parents[1] / "data/injected/pipeline_input.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Dataset preparado com {len(items)} registros em '{output_path}'")
