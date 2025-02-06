---
author: "Apiwat Ruangkanjanapaisarn"
title: "ทำความรู้จักกับ Agentic Design Patterns: รูปแบบการออกแบบ AI ที่ช่วยให้ระบบทำงานได้อย่างชาญฉลาด"
date: 2025-02-06
description: "เจาะลึกรูปแบบการออกแบบ AI ที่ช่วยให้ระบบ AI สามารถคิด ตัดสินใจ และทำงานได้อย่างอิสระ พร้อมตัวอย่างการประยุกต์ใช้งานในโลกจริง"
weight: 6
tags: ["AI", "Artificial Intelligence", "Agentic AI", "Design Patterns", "Software Engineering", "Python"]
categories: ["Artificial Intelligence"]
cover:
  image: https://raw.githubusercontent.com/ksm26/AI-Agentic-Design-Patterns-with-AutoGen/main/images/l1.png
  caption: "Image from (https://github.com/ksm26/AI-Agentic-Design-Patterns-with-AutoGen)"
---

ในยุคที่ AI กำลังเข้ามามีบทบาทสำคัญในชีวิตประจำวันของเรามากขึ้น การออกแบบระบบ AI ให้สามารถทำงานได้อย่างชาญฉลาดและมีประสิทธิภาพจึงเป็นเรื่องที่สำคัญมาก หนึ่งในแนวคิดที่น่าสนใจคือ "Agentic Design Patterns" หรือรูปแบบการออกแบบที่ช่วยให้ระบบ AI สามารถคิด ตัดสินใจ และทำงานได้อย่างอิสระ มาทำความรู้จักกับแนวคิดนี้กันให้ลึกซึ้งยิ่งขึ้น

## Agentic Design Patterns คืออะไร?

Agentic Design Patterns เป็นแนวทางการออกแบบที่ใช้ในการสร้างระบบ AI ที่สามารถทำงานได้อย่างอิสระ (Autonomous) โดยไม่ต้องพึ่งพาการควบคุมจากมนุษย์ตลอดเวลา รูปแบบการออกแบบเหล่านี้ช่วยกำหนดวิธีการที่ระบบ AI จะคิด ตัดสินใจ และมีปฏิสัมพันธ์กับสภาพแวดล้อม รวมถึงระบบอื่นๆ เพื่อให้บรรลุเป้าหมายที่ต้องการ

## รูปแบบการออกแบบที่น่าสนใจ

### 1. ReACT - การผสมผสานระหว่างการคิดและการกระทำ

ReACT (Reasoning and Acting) เป็นรูปแบบที่น่าสนใจมาก เพราะจำลองการทำงานคล้ายกับวิธีที่มนุษย์เราคิดและตัดสินใจ โดยระบบจะ:

1. วิเคราะห์สถานการณ์และคิดหาทางแก้ไข
2. ลงมือทำตามแผนที่วางไว้
3. ประเมินผลลัพธ์และปรับปรุงการตัดสินใจในรอบถัดไป

ตัวอย่างที่เห็นได้ชัดคือ ระบบวางแผนการเดินทาง ที่จะสลับไปมาระหว่างการค้นหาเที่ยวบิน (การคิด) และการจองตั๋ว (การกระทำ) โดยปรับเปลี่ยนแผนตามราคาและความพร้อมของเที่ยวบินที่พบ

> {{< collapse summary="***ตัวอย่าง ReACT***" >}}
```python
from typing import Dict, List, Tuple, TypedDict, Annotated
from datetime import datetime, timedelta
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import os

# Define our state structure
class AgentState(TypedDict):
    messages: List[str]
    current_plan: Dict
    flight_data: Dict
    booking_status: str
    next_action: str

# Mock flight database
MOCK_FLIGHTS = {
    "BKK-NRT": [
        {"flight": "TG676", "departure": "10:30", "arrival": "18:45", "price": 15000},
        {"flight": "JL708", "departure": "08:00", "arrival": "15:30", "price": 18000},
    ],
    "NRT-BKK": [
        {"flight": "TG677", "departure": "19:30", "arrival": "00:45", "price": 16000},
        {"flight": "JL709", "departure": "16:30", "arrival": "22:00", "price": 17000},
    ]
}

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",  # หรือใช้โมเดลอื่นที่ OpenRouter รองรับ
    temperature=0,
    headers={
        "HTTP-Referer": "http://localhost:8000",  # your website URL
        "X-Title": "Flight Booking Assistant"  # your app name
    }
)

def reasoning_step(state: AgentState) -> AgentState:
    """
    Analyze the current situation and plan next steps
    """
    # Create a prompt for the LLM to analyze the situation
    messages = [
        HumanMessage(content=f"""
        Current situation:
        - Planning status: {state['current_plan']}
        - Available flight data: {state['flight_data']}
        - Booking status: {state['booking_status']}
        
        What should be the next action? Choose from:
        1. SEARCH_FLIGHTS - If we need to look for flights
        2. BOOK_FLIGHT - If we found a suitable flight and should book it
        3. END - If we've completed the booking
        
        Provide your reasoning and the next action.
        """)
    ]
    
    # Get LLM response
    response = llm.invoke(messages)
    
    # Extract next action from response
    if "SEARCH_FLIGHTS" in response.content:
        state["next_action"] = "SEARCH_FLIGHTS"
    elif "BOOK_FLIGHT" in response.content:
        state["next_action"] = "BOOK_FLIGHT"
    elif "END" in response.content:
        state["next_action"] = "END"
    
    state["messages"].append(f"Reasoning: {response.content}")
    return state

def search_flights(state: AgentState) -> AgentState:
    """
    Search for available flights based on the current plan
    """
    route = f"{state['current_plan']['origin']}-{state['current_plan']['destination']}"
    if route in MOCK_FLIGHTS:
        state["flight_data"] = MOCK_FLIGHTS[route]
        state["messages"].append(f"Found {len(MOCK_FLIGHTS[route])} flights for {route}")
    else:
        state["flight_data"] = {}
        state["messages"].append(f"No flights found for {route}")
    
    return state

def book_flight(state: AgentState) -> AgentState:
    """
    Attempt to book the selected flight
    """
    if state["flight_data"]:
        # Find the cheapest flight
        selected_flight = min(state["flight_data"], key=lambda x: x["price"])
        state["booking_status"] = "CONFIRMED"
        state["messages"].append(
            f"Booked flight {selected_flight['flight']} for {selected_flight['price']} THB"
        )
    else:
        state["booking_status"] = "FAILED"
        state["messages"].append("Booking failed - no flights available")
    
    return state

def router(state: AgentState) -> str:
    """
    Route to the next step based on the reasoning outcome
    """
    return state["next_action"]

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("reasoning", reasoning_step)
workflow.add_node("search_flights", search_flights)
workflow.add_node("book_flight", book_flight)

# Add edges
workflow.add_edge("reasoning", router)
workflow.add_edge("search_flights", "reasoning")
workflow.add_edge("book_flight", "reasoning")

# Set entry point
workflow.set_entry_point("reasoning")

# Create conditional edges
workflow.add_conditional_edges(
    "reasoning",
    router,
    {
        "SEARCH_FLIGHTS": "search_flights",
        "BOOK_FLIGHT": "book_flight",
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "current_plan": {
            "origin": "BKK",
            "destination": "NRT",
            "date": "2025-02-15"
        },
        "flight_data": {},
        "booking_status": "NOT_STARTED",
        "next_action": "SEARCH_FLIGHTS"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 2. ระบบที่พัฒนาตัวเองได้ (Self-Improvement)

ความน่าสนใจของรูปแบบนี้อยู่ที่ความสามารถในการเรียนรู้และพัฒนาตัวเองอย่างต่อเนื่อง เหมือนกับที่มนุษย์เราเรียนรู้จากประสบการณ์ ระบบจะ:

- ประเมินผลการทำงานของตัวเอง
- เรียนรู้จากข้อมูลใหม่ๆ
- ปรับปรุงกระบวนการทำงานภายใน

ตัวอย่างที่เห็นได้บ่อยคือ ผู้ช่วยเขียนโค้ด ที่จะปรับปรุงคำแนะนำให้ดีขึ้นจากการวิเคราะห์ผลตอบรับของผู้ใช้

> {{< collapse summary="***ตัวอย่าง Self-Improvement***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import numpy as np
from dataclasses import dataclass, asdict
import os

# Define our state and data structures
@dataclass
class CodeSuggestion:
    code: str
    explanation: str
    confidence: float
    timestamp: str
    feedback_score: Optional[float] = None

class AssistantState(TypedDict):
    messages: List[str]
    current_query: str
    suggestions_history: List[Dict]
    feedback_history: List[Dict]
    improvement_metrics: Dict
    next_action: str
    current_suggestion: Optional[Dict]

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",  # หรือโมเดลอื่นที่ OpenRouter รองรับ
    temperature=0.7,
    headers={
        "HTTP-Referer": "http://localhost:8000",  # your website URL
        "X-Title": "Code Assistant"  # your app name
    }
)

# Knowledge base for code patterns (would be expanded in real implementation)
CODE_PATTERNS_DB = {
    "error_handling": {
        "weight": 0.5,
        "examples": ["try-except blocks", "error logging", "graceful degradation"]
    },
    "code_style": {
        "weight": 0.3,
        "examples": ["PEP8 compliance", "meaningful variable names", "proper indentation"]
    },
    "performance": {
        "weight": 0.2,
        "examples": ["algorithmic efficiency", "memory usage", "optimization techniques"]
    }
}

def generate_code_suggestion(state: AssistantState) -> AssistantState:
    """
    Generate code suggestions based on current knowledge and past feedback
    """
    messages = [
        HumanMessage(content=f"""
        Task: Generate Python code based on the following:
        Query: {state['current_query']}
        
        Consider past feedback patterns:
        {json.dumps(state['improvement_metrics'], indent=2)}
        
        Provide:
        1. Code suggestion (in ```python code blocks)
        2. Explanation:
        3. Confidence: (a number between 0-1)
        """)
    ]
    
    response = llm.invoke(messages)
    
    # Parse LLM response (in real implementation, would use more robust parsing)
    suggestion = CodeSuggestion(
        code=response.content.split("```python")[1].split("```")[0].strip(),
        explanation=response.content.split("Explanation:")[1].split("Confidence:")[0].strip(),
        confidence=float(response.content.split("Confidence:")[1].strip()),
        timestamp=datetime.now().isoformat()
    )
    
    state["current_suggestion"] = asdict(suggestion)
    state["suggestions_history"].append(asdict(suggestion))
    state["next_action"] = "EVALUATE"
    
    return state

def evaluate_feedback(state: AssistantState) -> AssistantState:
    """
    Analyze user feedback and update improvement metrics
    """
    if not state["feedback_history"]:
        state["next_action"] = "END"
        return state
    
    recent_feedback = state["feedback_history"][-1]
    
    # Analyze feedback and update metrics
    feedback_score = recent_feedback.get("score", 0)
    feedback_comments = recent_feedback.get("comments", "")
    
    # Update improvement metrics based on feedback
    messages = [
        HumanMessage(content=f"""
        Analyze this feedback and identify which patterns need adjustment:
        Score: {feedback_score}
        Comments: {feedback_comments}
        Current metrics: {json.dumps(state['improvement_metrics'], indent=2)}
        
        Analyze which patterns (error_handling, code_style, performance) are mentioned
        in the feedback and should be adjusted.
        """)
    ]
    
    response = llm.invoke(messages)
    
    # Update metrics (simplified version)
    for pattern in CODE_PATTERNS_DB:
        if pattern in response.content.lower():
            state["improvement_metrics"][pattern]["weight"] *= (1 + feedback_score * 0.1)
    
    # Normalize weights
    total_weight = sum(m["weight"] for m in state["improvement_metrics"].values())
    for pattern in state["improvement_metrics"]:
        state["improvement_metrics"][pattern]["weight"] /= total_weight
    
    state["next_action"] = "REFLECT"
    return state

def self_reflection(state: AssistantState) -> AssistantState:
    """
    Periodic self-reflection to identify areas for improvement
    """
    if len(state["suggestions_history"]) < 5:  # Need more data for meaningful reflection
        state["next_action"] = "END"
        return state
    
    # Analyze recent performance
    recent_suggestions = state["suggestions_history"][-5:]
    avg_confidence = np.mean([s["confidence"] for s in recent_suggestions])
    
    messages = [
        HumanMessage(content=f"""
        Analyze recent performance and suggest improvements:
        Average confidence: {avg_confidence}
        Recent suggestions: {json.dumps(recent_suggestions, indent=2)}
        Current metrics: {json.dumps(state['improvement_metrics'], indent=2)}
        
        Please provide specific suggestions for improving:
        1. Code quality
        2. Explanation clarity
        3. Confidence estimation accuracy
        """)
    ]
    
    response = llm.invoke(messages)
    
    # Update state with reflection insights
    state["messages"].append(f"Self-reflection insights: {response.content}")
    state["next_action"] = "END"
    
    return state

def router(state: AssistantState) -> str:
    """
    Route to the next step based on the current state
    """
    return state["next_action"]

# Create the graph
workflow = StateGraph(AssistantState)

# Add nodes
workflow.add_node("generate", generate_code_suggestion)
workflow.add_node("evaluate", evaluate_feedback)
workflow.add_node("reflect", self_reflection)

# Add edges
workflow.add_edge("generate", router)
workflow.add_edge("evaluate", router)
workflow.add_edge("reflect", router)

# Set entry point
workflow.set_entry_point("generate")

# Create conditional edges
workflow.add_conditional_edges(
    "generate",
    router,
    {
        "EVALUATE": "evaluate",
        "END": END
    }
)

workflow.add_conditional_edges(
    "evaluate",
    router,
    {
        "REFLECT": "reflect",
        "END": END
    }
)

workflow.add_conditional_edges(
    "reflect",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "current_query": "Write a function to find the nth Fibonacci number with memoization",
        "suggestions_history": [],
        "feedback_history": [
            {
                "score": 0.8,
                "comments": "Good use of memoization, but could improve error handling"
            }
        ],
        "improvement_metrics": {
            "error_handling": {"weight": 0.3, "count": 0},
            "code_style": {"weight": 0.4, "count": 0},
            "performance": {"weight": 0.3, "count": 0}
        },
        "next_action": "GENERATE",
        "current_suggestion": None
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 3. Agentic RAG - การผสมผสานการค้นหาและการสร้างเนื้อหา

รูปแบบนี้น่าสนใจเพราะช่วยให้ระบบ AI สามารถใช้ข้อมูลจากแหล่งภายนอกมาประกอบการตัดสินใจได้ โดย:

- ค้นหาข้อมูลที่เกี่ยวข้องจากฐานข้อมูล
- นำข้อมูลมาประมวลผลและสร้างเป็นคำตอบ
- ตรวจสอบความถูกต้องของข้อมูลก่อนนำไปใช้

ระบบแชทบอทให้บริการลูกค้าที่สามารถค้นหาข้อมูลจากเอกสารนโยบายและสร้างคำตอบที่เหมาะสมเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Agentic RAG***" >}}
```python
from typing import Dict, List, TypedDict, Optional
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass, asdict
import os

# Define data structures
@dataclass
class RetrievedDocument:
    content: str
    source: str
    relevance_score: float

@dataclass
class GeneratedResponse:
    response: str
    confidence: float
    sources: List[str]
    timestamp: str

class ChatbotState(TypedDict):
    messages: List[str]
    current_query: str
    retrieved_docs: List[Dict]
    generated_response: Optional[Dict]
    verification_result: Optional[Dict]
    next_action: str

# Mock policy database
POLICY_DOCUMENTS = {
    "returns": """
    Return Policy:
    - Items can be returned within 30 days of purchase
    - Must have original receipt
    - Items must be unused and in original packaging
    - Shipping costs are non-refundable
    - Store credit or refund will be issued within 7 business days
    """,
    
    "shipping": """
    Shipping Policy:
    - Free shipping on orders over 1000 THB
    - Standard shipping: 3-5 business days
    - Express shipping: 1-2 business days (additional fee)
    - International shipping available to select countries
    - Tracking number provided for all shipments
    """,
    
    "warranty": """
    Warranty Policy:
    - 1-year manufacturer warranty on all products
    - Covers defects in materials and workmanship
    - Does not cover damage from misuse or accidents
    - Warranty claims require proof of purchase
    - Replacement or repair decision at company discretion
    """
}

# Initialize components
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.7,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Policy Assistant"
    }
)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # หรือจะใช้โมเดลอื่นที่ Ollama รองรับ
    base_url="http://localhost:11434"  # ปรับ URL ตาม Ollama server
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Create vector store from policy documents
def initialize_vector_store():
    texts = []
    metadatas = []
    for policy_type, content in POLICY_DOCUMENTS.items():
        chunks = text_splitter.split_text(content)
        texts.extend(chunks)
        metadatas.extend([{"source": policy_type} for _ in chunks])
    
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

vector_store = initialize_vector_store()

def retrieve_relevant_docs(state: ChatbotState) -> ChatbotState:
    """
    Retrieve relevant documents based on the user query
    """
    # Search vector store
    docs = vector_store.similarity_search_with_score(
        state["current_query"],
        k=3
    )
    
    # Process and store retrieved documents
    retrieved_docs = [
        asdict(RetrievedDocument(
            content=doc[0].page_content,
            source=doc[0].metadata["source"],
            relevance_score=float(doc[1])
        ))
        for doc in docs
    ]
    
    state["retrieved_docs"] = retrieved_docs
    state["next_action"] = "GENERATE"
    
    return state

def generate_response(state: ChatbotState) -> ChatbotState:
    """
    Generate response using retrieved documents
    """
    # Prepare context from retrieved documents
    context = "\n".join([
        f"Document from {doc['source']}:\n{doc['content']}"
        for doc in state["retrieved_docs"]
    ])
    
    messages = [
        HumanMessage(content=f"""
        Based on the following context and user query, generate a helpful response.
        
        Context:
        {context}
        
        User Query:
        {state['current_query']}
        
        Generate a response that:
        1. Directly answers the user's question
        2. References specific policies when relevant
        3. Is clear and easy to understand
        4. Uses bullet points for important information
        """)
    ]
    
    response = llm.invoke(messages)
    
    generated_response = GeneratedResponse(
        response=response.content,
        confidence=0.8,  # In real implementation, would calculate based on relevance scores
        sources=[doc["source"] for doc in state["retrieved_docs"]],
        timestamp=datetime.now().isoformat()
    )
    
    state["generated_response"] = asdict(generated_response)
    state["next_action"] = "VERIFY"
    
    return state

def verify_response(state: ChatbotState) -> ChatbotState:
    """
    Verify the generated response against policy documents
    """
    response = state["generated_response"]["response"]
    sources = state["retrieved_docs"]
    
    messages = [
        HumanMessage(content=f"""
        Verify this response against the source documents for accuracy:
        
        Response:
        {response}
        
        Source Documents:
        {json.dumps(sources, indent=2)}
        
        Check for:
        1. Factual accuracy
        2. Completeness
        3. Consistency with policies
        4. Any missing important information
        
        Provide:
        1. Verification result (verified/not verified)
        2. Confidence score (0-1)
        3. Detailed notes about any issues found
        """)
    ]
    
    verification = llm.invoke(messages)
    
    state["verification_result"] = {
        "verified": "incorrect information" not in verification.content.lower(),
        "confidence": 0.9 if "high confidence" in verification.content.lower() else 0.7,
        "notes": verification.content
    }
    
    state["next_action"] = "END" if state["verification_result"]["verified"] else "GENERATE"
    
    return state

def router(state: ChatbotState) -> str:
    """
    Route to the next step based on the current state
    """
    return state["next_action"]

# Create the graph
workflow = StateGraph(ChatbotState)

# Add nodes
workflow.add_node("retrieve", retrieve_relevant_docs)
workflow.add_node("generate", generate_response)
workflow.add_node("verify", verify_response)

# Add edges
workflow.add_edge("retrieve", router)
workflow.add_edge("generate", router)
workflow.add_edge("verify", router)

# Set entry point
workflow.set_entry_point("retrieve")

# Create conditional edges
workflow.add_conditional_edges(
    "retrieve",
    router,
    {
        "GENERATE": "generate",
        "END": END
    }
)

workflow.add_conditional_edges(
    "generate",
    router,
    {
        "VERIFY": "verify",
        "END": END
    }
)

workflow.add_conditional_edges(
    "verify",
    router,
    {
        "GENERATE": "generate",
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "current_query": "What is your return policy for unused items?",
        "retrieved_docs": [],
        "generated_response": None,
        "verification_result": None,
        "next_action": "RETRIEVE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 4. Meta-Agent - ผู้จัดการระบบอัจฉริยะ

เปรียบเสมือนผู้จัดการโครงการที่คอยประสานงานระหว่างทีมย่อยต่างๆ Meta-Agent จะ:

- แบ่งงานให้ระบบย่อยที่เชี่ยวชาญเฉพาะด้าน
- ประสานงานให้ทุกส่วนทำงานสอดคล้องกัน
- ติดตามและควบคุมคุณภาพของงาน

ตัวอย่างเช่น ระบบบริหารโครงการที่แบ่งงานให้ระบบย่อยดูแลเรื่องการจัดตารางเวลา งบประมาณ และการรายงานผล

> {{< collapse summary="***ตัวอย่าง Meta-Agent***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime, timedelta
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from dataclasses import dataclass, asdict
import os

# Define data structures
@dataclass
class Task:
    id: str
    name: str
    description: str
    duration: int  # in days
    dependencies: List[str]
    assigned_to: str
    status: str
    estimated_cost: float

@dataclass
class ScheduleReport:
    start_date: str
    end_date: str
    critical_path: List[str]
    milestone_dates: Dict[str, str]

@dataclass
class BudgetReport:
    total_cost: float
    cost_breakdown: Dict[str, float]
    budget_status: str
    warnings: List[str]

class ProjectState(TypedDict):
    messages: List[str]
    tasks: List[Dict]
    schedule: Optional[Dict]
    budget: Optional[Dict]
    quality_metrics: Dict
    next_action: str
    current_request: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",  # your website URL
        "X-Title": "Project Management Assistant"  # your app name
    }
)

class SchedulingAgent:
    """Agent responsible for project scheduling"""
    
    @staticmethod
    def create_schedule(tasks: List[Task]) -> ScheduleReport:
        # Create schedule based on task dependencies and durations
        today = datetime.now()
        schedule = {}
        scheduled_tasks = set()
        
        def can_schedule(task):
            return all(dep in scheduled_tasks for dep in task.dependencies)
        
        current_date = today
        while len(scheduled_tasks) < len(tasks):
            for task in tasks:
                if task.id not in scheduled_tasks and can_schedule(task):
                    schedule[task.id] = {
                        'start': current_date.strftime('%Y-%m-%d'),
                        'end': (current_date + timedelta(days=task.duration)).strftime('%Y-%m-%d')
                    }
                    scheduled_tasks.add(task.id)
            current_date += timedelta(days=1)
        
        return ScheduleReport(
            start_date=today.strftime('%Y-%m-%d'),
            end_date=max(date['end'] for date in schedule.values()),
            critical_path=list(schedule.keys()),  # Simplified critical path
            milestone_dates=schedule
        )

class BudgetingAgent:
    """Agent responsible for budget management"""
    
    @staticmethod
    def analyze_budget(tasks: List[Task], schedule: ScheduleReport) -> BudgetReport:
        total_cost = sum(task.estimated_cost for task in tasks)
        
        # Calculate cost breakdown by category
        cost_breakdown = {}
        for task in tasks:
            category = task.assigned_to
            cost_breakdown[category] = cost_breakdown.get(category, 0) + task.estimated_cost
        
        # Determine budget status and warnings
        warnings = []
        if total_cost > 100000:  # Example budget threshold
            warnings.append("Project exceeds recommended budget")
        
        status = "GREEN" if not warnings else "YELLOW"
        
        return BudgetReport(
            total_cost=total_cost,
            cost_breakdown=cost_breakdown,
            budget_status=status,
            warnings=warnings
        )

class QualityAgent:
    """Agent responsible for quality control"""
    
    @staticmethod
    def assess_quality(state: ProjectState) -> Dict:
        messages = [
            HumanMessage(content=f"""
            Analyze the project quality and provide assessment:
            
            Schedule Information:
            {json.dumps(state['schedule'], indent=2)}
            
            Budget Information:
            {json.dumps(state['budget'], indent=2)}
            
            Task Information:
            {json.dumps(state['tasks'], indent=2)}
            
            Please provide:
            1. Quality score (0-100)
            2. Risk level assessment (LOW/MEDIUM/HIGH)
            3. Specific improvement recommendations
            4. Areas of concern, if any
            
            Format your response with clear sections and bullet points.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Parse LLM response (simplified)
        quality_metrics = {
            "quality_score": 85,  # Would be parsed from response
            "risk_level": "MEDIUM",
            "recommendations": response.content.split("\n")
        }
        
        return quality_metrics

def meta_agent_coordinator(state: ProjectState) -> ProjectState:
    """
    Main coordination function that delegates tasks to specialized agents
    """
    messages = [
        HumanMessage(content=f"""
        Analyze this project management request and determine the next action:
        
        Current Request: {state['current_request']}
        Current State: {json.dumps(state, indent=2)}
        
        Available actions:
        1. SCHEDULE - Create or update project schedule
        2. BUDGET - Analyze project budget and costs
        3. QUALITY - Assess project quality and risks
        4. END - Complete the workflow
        
        Consider:
        - Dependencies between tasks
        - Resource allocation
        - Timeline constraints
        - Budget requirements
        
        Provide your reasoning and recommended next action.
        """)
    ]
    
    response = llm.invoke(messages)
    
    # Determine next action based on LLM response
    if "SCHEDULE" in response.content:
        state["next_action"] = "SCHEDULE"
    elif "BUDGET" in response.content:
        state["next_action"] = "BUDGET"
    elif "QUALITY" in response.content:
        state["next_action"] = "QUALITY"
    else:
        state["next_action"] = "END"
    
    return state

def handle_scheduling(state: ProjectState) -> ProjectState:
    """Handle scheduling tasks"""
    tasks = [Task(**task) for task in state["tasks"]]
    schedule = SchedulingAgent.create_schedule(tasks)
    state["schedule"] = asdict(schedule)
    state["next_action"] = "BUDGET"
    return state

def handle_budgeting(state: ProjectState) -> ProjectState:
    """Handle budget analysis"""
    tasks = [Task(**task) for task in state["tasks"]]
    budget = BudgetingAgent.analyze_budget(tasks, ScheduleReport(**state["schedule"]))
    state["budget"] = asdict(budget)
    state["next_action"] = "QUALITY"
    return state

def handle_quality(state: ProjectState) -> ProjectState:
    """Handle quality assessment"""
    quality_metrics = QualityAgent.assess_quality(state)
    state["quality_metrics"] = quality_metrics
    state["next_action"] = "END"
    return state

def router(state: ProjectState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(ProjectState)

# Add nodes
workflow.add_node("coordinate", meta_agent_coordinator)
workflow.add_node("schedule", handle_scheduling)
workflow.add_node("budget", handle_budgeting)
workflow.add_node("quality", handle_quality)

# Add edges
workflow.add_edge("coordinate", router)
workflow.add_edge("schedule", router)
workflow.add_edge("budget", router)
workflow.add_edge("quality", router)

# Set entry point
workflow.set_entry_point("coordinate")

# Create conditional edges
workflow.add_conditional_edges(
    "coordinate",
    router,
    {
        "SCHEDULE": "schedule",
        "BUDGET": "budget",
        "QUALITY": "quality",
        "END": END
    }
)

workflow.add_conditional_edges(
    "schedule",
    router,
    {
        "BUDGET": "budget",
        "END": END
    }
)

workflow.add_conditional_edges(
    "budget",
    router,
    {
        "QUALITY": "quality",
        "END": END
    }
)

workflow.add_conditional_edges(
    "quality",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state with example project tasks
    initial_state = {
        "messages": [],
        "current_request": "Create project schedule and analyze budget",
        "tasks": [
            {
                "id": "T1",
                "name": "Requirements Analysis",
                "description": "Gather and analyze project requirements",
                "duration": 5,
                "dependencies": [],
                "assigned_to": "Analysis Team",
                "status": "Not Started",
                "estimated_cost": 20000
            },
            {
                "id": "T2",
                "name": "System Design",
                "description": "Create system architecture and design",
                "duration": 10,
                "dependencies": ["T1"],
                "assigned_to": "Design Team",
                "status": "Not Started",
                "estimated_cost": 30000
            },
            {
                "id": "T3",
                "name": "Implementation",
                "description": "Develop the system",
                "duration": 20,
                "dependencies": ["T2"],
                "assigned_to": "Development Team",
                "status": "Not Started",
                "estimated_cost": 50000
            }
        ],
        "schedule": None,
        "budget": None,
        "quality_metrics": {},
        "next_action": "COORDINATE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 5. ระบบวางแผนและปฏิบัติ (Planner-Executor)

รูปแบบนี้แยกการทำงานเป็นสองส่วนชัดเจน คือ:

ส่วนวางแผน:
- วิเคราะห์สถานการณ์
- กำหนดกลยุทธ์
- จัดลำดับความสำคัญของงาน

ส่วนปฏิบัติ:
- ดำเนินการตามแผน
- รายงานความคืบหน้า
- แจ้งเตือนเมื่อพบปัญหา

ระบบ AI ที่เล่นเกมเป็นตัวอย่างที่ดี โดยส่วนวางแผนจะคิดกลยุทธ์การเล่น และส่วนปฏิบัติจะควบคุมการเคลื่อนไหวในเกม

> {{< collapse summary="***ตัวอย่าง Planner-Executor***" >}}
```python
from typing import Dict, List, TypedDict, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import os

# Define game-specific structures
@dataclass
class GameState:
    board: List[List[str]]  # tic-tac-toe board
    current_player: str
    moves_made: int
    game_over: bool

@dataclass
class Strategy:
    main_goal: str
    priority_positions: List[Tuple[int, int]]
    fallback_positions: List[Tuple[int, int]]
    expected_outcome: str

@dataclass
class Move:
    position: Tuple[int, int]
    player: str
    priority: int
    reasoning: str

class AIState(TypedDict):
    messages: List[str]
    game_state: Dict
    current_strategy: Optional[Dict]
    planned_move: Optional[Dict]
    execution_result: Optional[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",  # หรือโมเดลอื่นที่ OpenRouter รองรับ
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",  # your website URL
        "X-Title": "Tic-Tac-Toe AI"  # your app name
    }
)

class GameManager:
    """Manages the game state and rules"""
    
    @staticmethod
    def create_empty_board() -> List[List[str]]:
        return [[" " for _ in range(3)] for _ in range(3)]
    
    @staticmethod
    def is_valid_move(board: List[List[str]], position: Tuple[int, int]) -> bool:
        row, col = position
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        return board[row][col] == " "
    
    @staticmethod
    def make_move(board: List[List[str]], position: Tuple[int, int], player: str) -> List[List[str]]:
        new_board = [row[:] for row in board]
        row, col = position
        new_board[row][col] = player
        return new_board
    
    @staticmethod
    def check_winner(board: List[List[str]]) -> Optional[str]:
        # Check rows
        for row in board:
            if row[0] == row[1] == row[2] != " ":
                return row[0]
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != " ":
                return board[0][col]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != " ":
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != " ":
            return board[0][2]
        
        return None

class Planner:
    """Strategic planner for the game"""
    
    @staticmethod
    def analyze_game_state(state: GameState) -> Strategy:
        board_str = "\n".join(["|".join(row) for row in state.board])
        
        messages = [
            HumanMessage(content=f"""
            Analyze this Tic-Tac-Toe board and create a strategic plan:
            
            Current Board:
            {board_str}
            
            Player: {state.current_player}
            Moves Made: {state.moves_made}
            
            Please provide a structured strategy with:
            1. Main Goal: [win/block/develop]
            2. Priority Positions: Provide coordinates as tuples (row, col), 0-2
            3. Fallback Positions: Provide alternative coordinates
            4. Expected Outcome: [Victory/Draw/Continue]
            
            Consider:
            - Winning opportunities
            - Blocking opponent's winning moves
            - Center and corner control
            - Development of winning patterns
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Parse response to create strategy (simplified)
        # In real implementation, would use more robust parsing
        strategy = Strategy(
            main_goal="Win in next move" if "win" in response.content.lower() else "Block opponent",
            priority_positions=[(1, 1), (0, 0), (2, 2)],  # Example positions
            fallback_positions=[(0, 1), (1, 0), (1, 2), (2, 1)],
            expected_outcome="Victory" if "victory" in response.content.lower() else "Continue"
        )
        
        return strategy

class Executor:
    """Tactical executor of planned moves"""
    
    @staticmethod
    def execute_move(game_state: GameState, strategy: Strategy) -> Move:
        # Try priority positions first
        for pos in strategy.priority_positions:
            if GameManager.is_valid_move(game_state.board, pos):
                return Move(
                    position=pos,
                    player=game_state.current_player,
                    priority=1,
                    reasoning="Priority position available"
                )
        
        # Try fallback positions
        for pos in strategy.fallback_positions:
            if GameManager.is_valid_move(game_state.board, pos):
                return Move(
                    position=pos,
                    player=game_state.current_player,
                    priority=2,
                    reasoning="Fallback position used"
                )
        
        # Find any valid move if nothing else is available
        for i in range(3):
            for j in range(3):
                if GameManager.is_valid_move(game_state.board, (i, j)):
                    return Move(
                        position=(i, j),
                        player=game_state.current_player,
                        priority=3,
                        reasoning="Last resort move"
                    )
        
        raise ValueError("No valid moves available")

def strategic_planning(state: AIState) -> AIState:
    """
    Strategic planning phase
    """
    game_state = GameState(**state["game_state"])
    strategy = Planner.analyze_game_state(game_state)
    
    state["current_strategy"] = asdict(strategy)
    state["next_action"] = "EXECUTE"
    
    return state

def tactical_execution(state: AIState) -> AIState:
    """
    Tactical execution phase
    """
    game_state = GameState(**state["game_state"])
    strategy = Strategy(**state["current_strategy"])
    
    # Execute the planned move
    move = Executor.execute_move(game_state, strategy)
    
    # Update game state with the executed move
    new_board = GameManager.make_move(
        game_state.board,
        move.position,
        game_state.current_player
    )
    
    # Update state
    state["game_state"]["board"] = new_board
    state["game_state"]["moves_made"] += 1
    state["execution_result"] = asdict(move)
    
    # Check for game end conditions
    winner = GameManager.check_winner(new_board)
    if winner or state["game_state"]["moves_made"] >= 9:
        state["game_state"]["game_over"] = True
        state["next_action"] = "END"
    else:
        state["next_action"] = "PLAN"
    
    return state

def router(state: AIState) -> str:
    """
    Route to the next step based on the current state
    """
    return state["next_action"]

# Create the graph
workflow = StateGraph(AIState)

# Add nodes
workflow.add_node("plan", strategic_planning)
workflow.add_node("execute", tactical_execution)

# Add edges
workflow.add_edge("plan", router)
workflow.add_edge("execute", router)

# Set entry point
workflow.set_entry_point("plan")

# Create conditional edges
workflow.add_conditional_edges(
    "plan",
    router,
    {
        "EXECUTE": "execute",
        "END": END
    }
)

workflow.add_conditional_edges(
    "execute",
    router,
    {
        "PLAN": "plan",
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state with empty game
    initial_state = {
        "messages": [],
        "game_state": {
            "board": GameManager.create_empty_board(),
            "current_player": "X",
            "moves_made": 0,
            "game_over": False
        },
        "current_strategy": None,
        "planned_move": None,
        "execution_result": None,
        "next_action": "PLAN"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 6. Reflexive Agent - ระบบตอบสนองอัตโนมัติ

รูปแบบนี้เน้นการตอบสนองที่รวดเร็วต่อการเปลี่ยนแปลง โดย:

- ตรวจจับการเปลี่ยนแปลงในสภาพแวดล้อม
- ตอบสนองทันทีตามกฎที่กำหนดไว้
- ไม่ต้องใช้เวลาคิดวิเคราะห์มาก

หุ่นยนต์ดูดฝุ่นที่หลบสิ่งกีดขวางอัตโนมัติเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Reflexive Agent***" >}}
```python
from typing import Dict, List, TypedDict, Tuple, Optional
from enum import Enum
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenRouter
from langchain_core.messages import HumanMessage, AIMessage
import random
import os

# Define enums for direction and action
class Direction(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

class Action(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    CLEAN = "CLEAN"
    RETURN_HOME = "RETURN_HOME"

# Define data structures
@dataclass
class Position:
    x: int
    y: int

@dataclass
class SensorData:
    front_obstacle: bool
    left_obstacle: bool
    right_obstacle: bool
    dirt_detected: bool
    battery_level: int

@dataclass
class RobotState:
    position: Position
    direction: Direction
    battery_level: int
    cleaned_positions: List[Tuple[int, int]]

class AgentState(TypedDict):
    messages: List[str]
    robot_state: Dict
    sensor_data: Dict
    action_taken: Optional[str]
    next_action: str

# Simulated environment
class Environment:
    def __init__(self, size: int = 10):
        self.size = size
        self.obstacles = self._generate_obstacles()
        self.dirt_locations = self._generate_dirt()
        
    def _generate_obstacles(self) -> List[Tuple[int, int]]:
        # Generate random obstacles
        obstacles = []
        num_obstacles = self.size * 2
        for _ in range(num_obstacles):
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x, y) != (0, 0):  # Keep starting position clear
                obstacles.append((x, y))
        return obstacles
    
    def _generate_dirt(self) -> List[Tuple[int, int]]:
        # Generate random dirt locations
        dirt = []
        num_dirt = self.size * 3
        for _ in range(num_dirt):
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x, y) not in self.obstacles:
                dirt.append((x, y))
        return dirt
    
    def get_sensor_data(self, position: Position, direction: Direction) -> SensorData:
        # Check for obstacles in adjacent positions
        front_pos = self._get_adjacent_position(position, direction)
        left_pos = self._get_adjacent_position(position, self._turn_left(direction))
        right_pos = self._get_adjacent_position(position, self._turn_right(direction))
        
        return SensorData(
            front_obstacle=self._is_obstacle(front_pos),
            left_obstacle=self._is_obstacle(left_pos),
            right_obstacle=self._is_obstacle(right_pos),
            dirt_detected=(position.x, position.y) in self.dirt_locations,
            battery_level=100  # Simplified battery simulation
        )
    
    def _is_obstacle(self, position: Position) -> bool:
        # Check if position is obstacle or out of bounds
        if (position.x < 0 or position.x >= self.size or 
            position.y < 0 or position.y >= self.size):
            return True
        return (position.x, position.y) in self.obstacles
    
    @staticmethod
    def _get_adjacent_position(pos: Position, direction: Direction) -> Position:
        if direction == Direction.NORTH:
            return Position(pos.x, pos.y + 1)
        elif direction == Direction.SOUTH:
            return Position(pos.x, pos.y - 1)
        elif direction == Direction.EAST:
            return Position(pos.x + 1, pos.y)
        else:  # WEST
            return Position(pos.x - 1, pos.y)
    
    @staticmethod
    def _turn_left(direction: Direction) -> Direction:
        turns = {
            Direction.NORTH: Direction.WEST,
            Direction.WEST: Direction.SOUTH,
            Direction.SOUTH: Direction.EAST,
            Direction.EAST: Direction.NORTH
        }
        return turns[direction]
    
    @staticmethod
    def _turn_right(direction: Direction) -> Direction:
        turns = {
            Direction.NORTH: Direction.EAST,
            Direction.EAST: Direction.SOUTH,
            Direction.SOUTH: Direction.WEST,
            Direction.WEST: Direction.NORTH
        }
        return turns[direction]

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Robot Vacuum AI"
    }
)

class StrategicAnalyzer:
    """Analyzes environment and suggests optimal strategies"""
    
    @staticmethod
    def analyze_situation(
        sensor_data: SensorData,
        robot_state: RobotState,
        env_size: int
    ) -> Dict:
        # Create a map representation
        map_data = [["?" for _ in range(env_size)] for _ in range(env_size)]
        map_data[robot_state.position.y][robot_state.position.x] = "R"
        
        for x, y in robot_state.cleaned_positions:
            if map_data[y][x] != "R":
                map_data[y][x] = "C"
        
        map_str = "\n".join([" ".join(row) for row in map_data])
        
        messages = [
            HumanMessage(content=f"""
            Analyze the robot vacuum's situation and suggest a strategy:
            
            Current Map (R=Robot, C=Cleaned, ?=Unknown):
            {map_str}
            
            Sensor Data:
            - Front obstacle: {sensor_data.front_obstacle}
            - Left obstacle: {sensor_data.left_obstacle}
            - Right obstacle: {sensor_data.right_obstacle}
            - Dirt detected: {sensor_data.dirt_detected}
            - Battery level: {sensor_data.battery_level}%
            
            Robot State:
            - Position: ({robot_state.position.x}, {robot_state.position.y})
            - Direction: {robot_state.direction}
            - Cleaned positions: {len(robot_state.cleaned_positions)}
            
            Suggest:
            1. Primary objective
            2. Movement pattern
            3. Risk assessment
            4. Efficiency recommendations
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Parse LLM response (simplified)
        return {
            "strategy": response.content,
            "risk_level": "LOW" if sensor_data.battery_level > 50 else "MEDIUM",
            "recommended_pattern": "SPIRAL" if "spiral" in response.content.lower() else "ZIGZAG"
        }

class ReflexiveRules:
    """Define reflexive rules for the robot vacuum"""
    
    def __init__(self):
        self.analyzer = StrategicAnalyzer()
    
    def determine_action(
        self,
        sensor_data: SensorData,
        robot_state: RobotState,
        env_size: int
    ) -> Tuple[Action, str]:
        # Get strategic analysis for complex decisions
        analysis = self.analyzer.analyze_situation(sensor_data, robot_state, env_size)
        
        # Priority 1: Handle low battery
        if sensor_data.battery_level < 20:
            return Action.RETURN_HOME, "Low battery, returning to charging station"
        
        # Priority 2: Clean dirt if detected
        if sensor_data.dirt_detected:
            return Action.CLEAN, "Dirt detected, cleaning current position"
        
        # Priority 3: Navigate based on strategic analysis and obstacles
        if "SPIRAL" in analysis["recommended_pattern"]:
            if not sensor_data.right_obstacle:
                return Action.TURN_RIGHT, "Following spiral pattern"
            elif not sensor_data.front_obstacle:
                return Action.MOVE_FORWARD, "Following spiral pattern"
        
        # Default obstacle avoidance
        if sensor_data.front_obstacle:
            if not sensor_data.right_obstacle:
                return Action.TURN_RIGHT, f"Avoiding obstacle: {analysis['strategy']}"
            elif not sensor_data.left_obstacle:
                return Action.TURN_LEFT, f"Avoiding obstacle: {analysis['strategy']}"
            else:
                return Action.TURN_LEFT, "Surrounded by obstacles, turning around"
        
        # Default: Move forward
        return Action.MOVE_FORWARD, f"Following {analysis['recommended_pattern']} pattern"

def sense_and_act(state: AgentState) -> AgentState:
    """
    Main function for the reflexive agent - sense environment and act immediately
    """
    # Get current state
    robot_state = RobotState(**state["robot_state"])
    sensor_data = SensorData(**state["sensor_data"])
    
    # Create rules engine and determine action
    rules = ReflexiveRules()
    action, reasoning = rules.determine_action(sensor_data, robot_state, 10)  # 10 is env_size
    
    # Execute action and update state
    if action == Action.MOVE_FORWARD:
        new_position = Environment._get_adjacent_position(
            robot_state.position, 
            robot_state.direction
        )
        if not Environment._is_obstacle(Position(new_position.x, new_position.y)):
            robot_state.position = new_position
    
    elif action == Action.TURN_LEFT:
        robot_state.direction = Environment._turn_left(robot_state.direction)
    
    elif action == Action.TURN_RIGHT:
        robot_state.direction = Environment._turn_right(robot_state.direction)
    
    elif action == Action.CLEAN:
        pos = (robot_state.position.x, robot_state.position.y)
        if pos not in robot_state.cleaned_positions:
            robot_state.cleaned_positions.append(pos)
    
    # Update state
    state["robot_state"] = asdict(robot_state)
    state["action_taken"] = action
    state["messages"].append(reasoning)
    
    # Determine if we should continue or end
    if action == Action.RETURN_HOME and (robot_state.position.x, robot_state.position.y) == (0, 0):
        state["next_action"] = "END"
    else:
        state["next_action"] = "CONTINUE"
    
    return state

def router(state: AgentState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(AgentState)

# Add node
workflow.add_node("sense_and_act", sense_and_act)

# Add edge
workflow.add_edge("sense_and_act", router)

# Set entry point
workflow.set_entry_point("sense_and_act")

# Create conditional edges
workflow.add_conditional_edges(
    "sense_and_act",
    router,
    {
        "CONTINUE": "sense_and_act",
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize environment
    env = Environment(size=10)
    
    # Initialize state
    initial_state = {
        "messages": [],
        "robot_state": {
            "position": asdict(Position(0, 0)),
            "direction": Direction.NORTH,
            "battery_level": 100,
            "cleaned_positions": []
        },
        "sensor_data": asdict(env.get_sensor_data(Position(0, 0), Direction.NORTH)),
        "action_taken": None,
        "next_action": "CONTINUE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
        
        # Update sensor data for next iteration
        robot_state = RobotState(**output["robot_state"])
        output["sensor_data"] = asdict(env.get_sensor_data(
            Position(**robot_state.position),
            robot_state.direction
        ))
```
{{</ collapse >}}

### 7. Interactive Learning - การเรียนรู้แบบมีปฏิสัมพันธ์

รูปแบบนี้น่าสนใจเพราะช่วยให้ระบบพัฒนาได้จากการมีปฏิสัมพันธ์กับผู้ใช้ โดย:

- รับข้อเสนอแนะจากผู้ใช้
- วิเคราะห์และเรียนรู้จากข้อมูลป้อนกลับ
- ปรับปรุงพฤติกรรมให้ตรงกับความต้องการ

ระบบแปลภาษาที่เรียนรู้จากการแก้ไขของผู้ใช้เป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Interactive Learning***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import numpy as np
from collections import defaultdict
import os

# Define data structures (dataclasses remain the same)
@dataclass
class Translation:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    timestamp: str

@dataclass
class UserFeedback:
    translation_id: str
    corrected_text: str
    feedback_type: str  # GRAMMAR, CONTEXT, STYLE, etc.
    comment: str
    timestamp: str

@dataclass
class LearningPattern:
    pattern_type: str
    original_phrase: str
    corrected_phrase: str
    context: str
    frequency: int
    confidence: float

class TranslatorState(TypedDict):
    messages: List[str]
    current_text: str
    source_language: str
    target_language: str
    translation_history: List[Dict]
    feedback_history: List[Dict]
    learning_patterns: Dict[str, List[Dict]]
    current_translation: Optional[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.3,
    headers={
        "HTTP-Referer": "http://localhost:8000",  # your website URL
        "X-Title": "Translation Assistant"  # your app name
    }
)

class TranslationMemory:
    """Manages translation patterns and corrections"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        
    def add_pattern(self, pattern: LearningPattern):
        key = f"{pattern.pattern_type}:{pattern.original_phrase}"
        existing = next(
            (p for p in self.patterns[key] if p["corrected_phrase"] == pattern.corrected_phrase),
            None
        )
        
        if existing:
            existing["frequency"] += 1
            existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
        else:
            self.patterns[key].append(asdict(pattern))
    
    def get_relevant_patterns(self, text: str, pattern_type: str) -> List[Dict]:
        relevant = []
        for key, patterns in self.patterns.items():
            if key.startswith(f"{pattern_type}:"):
                original = key.split(":", 1)[1]
                if original.lower() in text.lower():
                    relevant.extend(patterns)
        return relevant

class FeedbackAnalyzer:
    """Analyzes user feedback to extract learning patterns"""
    
    @staticmethod
    def analyze_feedback(
        original: Translation,
        feedback: UserFeedback
    ) -> List[LearningPattern]:
        messages = [
            HumanMessage(content=f"""
            Analyze this translation pair and identify improvement patterns:
            
            Original Text: {original.original_text}
            Initial Translation: {original.translated_text}
            Corrected Translation: {feedback.corrected_text}
            Feedback Type: {feedback.feedback_type}
            User Comment: {feedback.comment}
            
            Please identify patterns in these categories:
            1. GRAMMAR: Grammar rules and structures
            2. CONTEXT: Context-specific word choices
            3. STYLE: Writing style and natural expression
            
            For each identified pattern, provide:
            - Pattern type (GRAMMAR/CONTEXT/STYLE)
            - Original phrase
            - Corrected phrase
            - Explanation of the improvement
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Extract patterns (simplified version)
        patterns = []
        for feedback_type in ["GRAMMAR", "CONTEXT", "STYLE"]:
            if feedback_type.lower() in response.content.lower():
                pattern = LearningPattern(
                    pattern_type=feedback_type,
                    original_phrase=original.translated_text,
                    corrected_phrase=feedback.corrected_text,
                    context=original.original_text,
                    frequency=1,
                    confidence=0.5
                )
                patterns.append(pattern)
        
        return patterns

def translate_with_memory(state: TranslatorState) -> TranslatorState:
    """
    Translate text using learned patterns and translation memory
    """
    memory = TranslationMemory()
    
    # Load patterns from state
    for pattern_type, patterns in state["learning_patterns"].items():
        for pattern in patterns:
            memory.add_pattern(LearningPattern(**pattern))
    
    # Get relevant patterns
    relevant_patterns = []
    for pattern_type in ["GRAMMAR", "CONTEXT", "STYLE"]:
        relevant_patterns.extend(
            memory.get_relevant_patterns(state["current_text"], pattern_type)
        )
    
    # Create translation prompt with patterns
    messages = [
        HumanMessage(content=f"""
        Translate the following text from {state['source_language']} to {state['target_language']}.
        
        Text to translate: {state['current_text']}
        
        Consider these learned patterns:
        {json.dumps(relevant_patterns, indent=2)}
        
        Please provide:
        1. Translation: (your translation)
        2. Confidence: (score between 0-1)
        3. Applied Patterns: (list which patterns were used)
        
        Format your response with clear sections.
        """)
    ]
    
    response = llm.invoke(messages)
    
    # Create translation object (with improved parsing)
    translation = Translation(
        original_text=state["current_text"],
        translated_text=response.content.split("Translation:")[1].split("Confidence:")[0].strip(),
        source_language=state["source_language"],
        target_language=state["target_language"],
        confidence_score=float(response.content.split("Confidence:")[1].split("\n")[0].strip()),
        timestamp=datetime.now().isoformat()
    )
    
    state["current_translation"] = asdict(translation)
    state["translation_history"].append(asdict(translation))
    state["next_action"] = "AWAIT_FEEDBACK"
    
    return state

def process_feedback(state: TranslatorState) -> TranslatorState:
    """
    Process user feedback and update learning patterns
    """
    if not state["feedback_history"]:
        state["next_action"] = "END"
        return state
    
    # Get latest feedback
    feedback = UserFeedback(**state["feedback_history"][-1])
    original = Translation(**state["current_translation"])
    
    # Analyze feedback
    analyzer = FeedbackAnalyzer()
    new_patterns = analyzer.analyze_feedback(original, feedback)
    
    # Update learning patterns
    memory = TranslationMemory()
    for pattern in new_patterns:
        pattern_type = pattern.pattern_type.lower()
        if pattern_type not in state["learning_patterns"]:
            state["learning_patterns"][pattern_type] = []
        memory.add_pattern(pattern)
        state["learning_patterns"][pattern_type].append(asdict(pattern))
    
    state["next_action"] = "END"
    return state

def process_feedback(state: TranslatorState) -> TranslatorState:
    """
    Process user feedback and update learning patterns
    """
    if not state["feedback_history"]:
        state["next_action"] = "END"
        return state
    
    # Get latest feedback
    feedback = UserFeedback(**state["feedback_history"][-1])
    original = Translation(**state["current_translation"])
    
    # Analyze feedback
    analyzer = FeedbackAnalyzer()
    new_patterns = analyzer.analyze_feedback(original, feedback)
    
    # Update learning patterns
    memory = TranslationMemory()
    for pattern in new_patterns:
        pattern_type = pattern.pattern_type.lower()
        if pattern_type not in state["learning_patterns"]:
            state["learning_patterns"][pattern_type] = []
        memory.add_pattern(pattern)
        state["learning_patterns"][pattern_type].append(asdict(pattern))
    
    state["next_action"] = "END"
    return state

def router(state: TranslatorState) -> str:
    """
    Route to the next step based on the current state
    """
    return state["next_action"]

# Create the graph
workflow = StateGraph(TranslatorState)

# Add nodes
workflow.add_node("translate", translate_with_memory)
workflow.add_node("feedback", process_feedback)

# Add edges
workflow.add_edge("translate", router)
workflow.add_edge("feedback", router)

# Set entry point
workflow.set_entry_point("translate")

# Create conditional edges
workflow.add_conditional_edges(
    "translate",
    router,
    {
        "AWAIT_FEEDBACK": "feedback",
        "END": END
    }
)

workflow.add_conditional_edges(
    "feedback",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "current_text": "The cat sat on the mat.",
        "source_language": "English",
        "target_language": "Thai",
        "translation_history": [],
        "feedback_history": [
            {
                "translation_id": "1",
                "corrected_text": "แมวนั่งอยู่บนเสื่อ",
                "feedback_type": "STYLE",
                "comment": "More natural Thai expression",
                "timestamp": datetime.now().isoformat()
            }
        ],
        "learning_patterns": {},
        "current_translation": None,
        "next_action": "TRANSLATE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 8. การแยกงานเป็นลำดับชั้น (Hierarchical Task Decomposition)

รูปแบบนี้ช่วยจัดการงานที่ซับซ้อนได้อย่างมีประสิทธิภาพ โดย:

- แยกงานใหญ่เป็นงานย่อยที่จัดการได้ง่ายขึ้น
- จัดลำดับความสำคัญของงานย่อย
- ติดตามความคืบหน้าในแต่ละระดับ

ผู้ช่วย AI ที่ช่วยจัดงานอีเวนต์ โดยแบ่งเป็นการจองสถานที่ ส่งการ์ดเชิญ และจัดตารางงาน เป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Hierarchical Task Decomposition***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import os

# Define data structures
@dataclass
class Task:
    id: str
    name: str
    description: str
    priority: int  # 1 (highest) to 5 (lowest)
    status: str    # NOT_STARTED, IN_PROGRESS, COMPLETED
    parent_id: Optional[str]
    subtasks: List[str]
    dependencies: List[str]
    assigned_to: str
    deadline: str
    progress: int  # 0-100%

@dataclass
class TaskUpdate:
    task_id: str
    status: str
    progress: int
    notes: str
    timestamp: str

class PlannerState(TypedDict):
    messages: List[str]
    event_details: Dict
    tasks: Dict[str, Dict]
    updates: List[Dict]
    current_focus: Optional[str]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",  # your website URL
        "X-Title": "Event Planning Assistant"  # your app name
    }
)

class TaskDecomposer:
    """Handles breaking down complex tasks into subtasks"""
    
    @staticmethod
    def decompose_event_planning(event_details: Dict) -> List[Task]:
        messages = [
            HumanMessage(content=f"""
            Please decompose this event into a structured task hierarchy:
            
            EVENT DETAILS:
            {json.dumps(event_details, indent=2)}
            
            MAIN CATEGORIES:
            1. Venue Management
            2. Guest Management
            3. Logistics & Schedule
            4. Budget & Contracts
            
            For each task, provide:
            1. Task ID (unique identifier)
            2. Task Name
            3. Description (clear and actionable)
            4. Priority (1-5, where 1 is highest)
            5. Dependencies (list of task IDs)
            6. Timeline and deadlines
            7. Team assignment
            
            Ensure tasks are:
            - Well-defined and measurable
            - Properly sequenced with dependencies
            - Assigned appropriate priorities
            - Given realistic timelines
            
            Format each task in a structured way.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create task hierarchy (simplified version - would parse LLM response in full implementation)
        tasks = [
            Task(
                id="VENUE_MAIN",
                name="Venue Management",
                description="Overall venue selection and management",
                priority=1,
                status="NOT_STARTED",
                parent_id=None,
                subtasks=["VENUE_SEARCH", "VENUE_BOOK", "VENUE_SETUP"],
                dependencies=[],
                assigned_to="Venue Team",
                deadline=(datetime.now() + timedelta(days=30)).isoformat(),
                progress=0
            ),
            Task(
                id="VENUE_SEARCH",
                name="Venue Search",
                description="Research and visit potential venues",
                priority=1,
                status="NOT_STARTED",
                parent_id="VENUE_MAIN",
                subtasks=[],
                dependencies=[],
                assigned_to="Venue Team",
                deadline=(datetime.now() + timedelta(days=7)).isoformat(),
                progress=0
            ),
            # Additional tasks would be parsed from LLM response
        ]
        
        return tasks

class TaskPrioritizer:
    """Manages task priorities and dependencies"""
    
    @staticmethod
    def calculate_critical_path(tasks: Dict[str, Task]) -> List[str]:
        critical_tasks = []
        remaining_tasks = set(tasks.keys())
        
        while remaining_tasks:
            for task_id in list(remaining_tasks):
                task = tasks[task_id]
                if all(dep not in remaining_tasks for dep in task.dependencies):
                    critical_tasks.append(task_id)
                    remaining_tasks.remove(task_id)
        
        return critical_tasks

    @staticmethod
    def update_priorities(tasks: Dict[str, Task]) -> Dict[str, Task]:
        critical_path = TaskPrioritizer.calculate_critical_path(tasks)
        
        # Adjust priorities based on critical path
        for i, task_id in enumerate(critical_path):
            tasks[task_id].priority = min(tasks[task_id].priority, 1 + i // 3)
        
        return tasks

class ProgressTracker:
    """Tracks and updates task progress"""
    
    @staticmethod
    def update_task_progress(
        tasks: Dict[str, Task],
        update: TaskUpdate
    ) -> Dict[str, Task]:
        if update.task_id not in tasks:
            return tasks
        
        # Update specific task
        task = tasks[update.task_id]
        task.status = update.status
        task.progress = update.progress
        
        # Update parent task progress
        if task.parent_id:
            parent = tasks[task.parent_id]
            subtask_progress = [
                tasks[subtask_id].progress
                for subtask_id in parent.subtasks
            ]
            parent.progress = sum(subtask_progress) // len(subtask_progress)
        
        return tasks

    @staticmethod
    def get_status_report(tasks: Dict[str, Task]) -> Dict:
        return {
            "total_tasks": len(tasks),
            "completed": sum(1 for t in tasks.values() if t.status == "COMPLETED"),
            "in_progress": sum(1 for t in tasks.values() if t.status == "IN_PROGRESS"),
            "not_started": sum(1 for t in tasks.values() if t.status == "NOT_STARTED"),
            "overall_progress": sum(t.progress for t in tasks.values()) // len(tasks)
        }

def decompose_tasks(state: PlannerState) -> PlannerState:
    """
    Initial task decomposition
    """
    decomposer = TaskDecomposer()
    tasks = decomposer.decompose_event_planning(state["event_details"])
    
    # Convert tasks to dictionary format
    state["tasks"] = {task.id: asdict(task) for task in tasks}
    state["next_action"] = "PRIORITIZE"
    
    return state

def prioritize_tasks(state: PlannerState) -> PlannerState:
    """
    Prioritize tasks and calculate critical path
    """
    # Convert dict back to Task objects
    tasks = {
        task_id: Task(**task_data)
        for task_id, task_data in state["tasks"].items()
    }
    
    # Update priorities
    prioritizer = TaskPrioritizer()
    tasks = prioritizer.update_priorities(tasks)
    
    # Convert back to dict format
    state["tasks"] = {task_id: asdict(task) for task_id, task in tasks.items()}
    state["next_action"] = "UPDATE_PROGRESS"
    
    return state

def update_progress(state: PlannerState) -> PlannerState:
    """
    Update task progress based on recent updates
    """
    if not state["updates"]:
        state["next_action"] = "END"
        return state
    
    # Convert dict back to Task objects
    tasks = {
        task_id: Task(**task_data)
        for task_id, task_data in state["tasks"].items()
    }
    
    # Process each update
    tracker = ProgressTracker()
    for update_data in state["updates"]:
        update = TaskUpdate(**update_data)
        tasks = tracker.update_task_progress(tasks, update)
    
    # Get status report
    status_report = tracker.get_status_report(tasks)
    
    # Update state
    state["tasks"] = {task_id: asdict(task) for task_id, task in tasks.items()}
    state["status_report"] = status_report
    state["next_action"] = "END"
    
    return state

def router(state: PlannerState) -> str:
    """
    Route to the next step based on the current state
    """
    return state["next_action"]

# Create the graph
workflow = StateGraph(PlannerState)

# Add nodes
workflow.add_node("decompose", decompose_tasks)
workflow.add_node("prioritize", prioritize_tasks)
workflow.add_node("progress", update_progress)

# Add edges
workflow.add_edge("decompose", router)
workflow.add_edge("prioritize", router)
workflow.add_edge("progress", router)

# Set entry point
workflow.set_entry_point("decompose")

# Create conditional edges
workflow.add_conditional_edges(
    "decompose",
    router,
    {
        "PRIORITIZE": "prioritize",
        "END": END
    }
)

workflow.add_conditional_edges(
    "prioritize",
    router,
    {
        "UPDATE_PROGRESS": "progress",
        "END": END
    }
)

workflow.add_conditional_edges(
    "progress",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state with example event
    initial_state = {
        "messages": [],
        "event_details": {
            "name": "Tech Conference 2025",
            "date": "2025-06-15",
            "expected_attendees": 200,
            "type": "Conference",
            "budget": 50000,
            "location_preference": "City Center",
            "special_requirements": ["AV Equipment", "Catering", "Registration Desk"]
        },
        "tasks": {},
        "updates": [
            {
                "task_id": "VENUE_SEARCH",
                "status": "COMPLETED",
                "progress": 100,
                "notes": "Selected Convention Center A",
                "timestamp": datetime.now().isoformat()
            }
        ],
        "current_focus": None,
        "next_action": "DECOMPOSE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 9. ระบบที่ทำงานตามเป้าหมาย (Goal-Oriented Agent)

รูปแบบนี้เน้นการทำงานที่มีจุดมุ่งหมายชัดเจน โดย:

- กำหนดเป้าหมายที่ต้องการ
- วางแผนการทำงานเพื่อให้บรรลุเป้าหมาย
- ปรับเปลี่ยนกลยุทธ์ตามสถานการณ์

ระบบวางแผนการเงินที่ปรับกลยุทธ์การลงทุนเพื่อให้บรรลุเป้าหมายการออมเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Goal-Oriented Agent***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import os

# Define data structures
@dataclass
class FinancialGoal:
    id: str
    name: str
    target_amount: float
    current_amount: float
    target_date: str
    priority: int  # 1 (highest) to 5 (lowest)
    risk_tolerance: str  # LOW, MEDIUM, HIGH
    progress: float  # Percentage

@dataclass
class Investment:
    type: str  # STOCKS, BONDS, SAVINGS, etc.
    amount: float
    expected_return: float
    risk_level: str
    liquidity: str
    allocation_percentage: float

@dataclass
class Strategy:
    goal_id: str
    investments: List[Investment]
    monthly_contribution: float
    expected_timeline: int  # months
    risk_assessment: str
    contingency_plans: List[str]

class PlannerState(TypedDict):
    messages: List[str]
    financial_goals: Dict[str, Dict]
    current_portfolio: Dict[str, float]
    market_conditions: Dict[str, str]
    strategies: Dict[str, Dict]
    analysis_results: Optional[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Financial Planning Assistant"
    }
)

class GoalAnalyzer:
    """Analyzes financial goals and current progress"""
    
    @staticmethod
    def analyze_goal_progress(goal: FinancialGoal, strategy: Strategy) -> Dict:
        messages = [
            HumanMessage(content=f"""
            Analyze progress towards this financial goal:
            
            Goal Details:
            - Name: {goal.name}
            - Target: ${goal.target_amount:,.2f}
            - Current: ${goal.current_amount:,.2f}
            - Deadline: {goal.target_date}
            - Risk Tolerance: {goal.risk_tolerance}
            
            Current Strategy:
            - Monthly Contribution: ${strategy.monthly_contribution:,.2f}
            - Timeline: {strategy.expected_timeline} months
            - Investment Mix: {json.dumps([asdict(inv) for inv in strategy.investments], indent=2)}
            
            Please analyze:
            1. Progress status and likelihood of achievement
            2. Risk alignment
            3. Required adjustments
            4. Recommendations for optimization
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Parse response for analysis results
        return {
            "status": "ON_TRACK" if goal.progress >= (goal.current_amount / goal.target_amount * 100) else "NEEDS_ADJUSTMENT",
            "analysis": response.content,
            "requires_strategy_update": "adjustment" in response.content.lower()
        }

class StrategyPlanner:
    """Plans and adjusts investment strategies"""
    
    @staticmethod
    def create_strategy(
        goal: FinancialGoal,
        current_portfolio: Dict[str, float],
        market_conditions: Dict[str, str]
    ) -> Strategy:
        messages = [
            HumanMessage(content=f"""
            Create an investment strategy for this financial goal:
            
            Goal:
            {json.dumps(asdict(goal), indent=2)}
            
            Current Portfolio:
            {json.dumps(current_portfolio, indent=2)}
            
            Market Conditions:
            {json.dumps(market_conditions, indent=2)}
            
            Provide a comprehensive strategy including:
            1. Asset allocation
            2. Monthly contribution requirements
            3. Risk management approach
            4. Timeline and milestones
            5. Contingency plans
            
            Consider:
            - Risk tolerance level
            - Time horizon
            - Current market conditions
            - Liquidity needs
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create strategy (simplified version - would parse LLM response in real implementation)
        return Strategy(
            goal_id=goal.id,
            investments=[
                Investment(
                    type="STOCKS",
                    amount=goal.target_amount * 0.6,
                    expected_return=0.08,
                    risk_level="MEDIUM",
                    liquidity="MEDIUM",
                    allocation_percentage=60
                ),
                Investment(
                    type="BONDS",
                    amount=goal.target_amount * 0.3,
                    expected_return=0.04,
                    risk_level="LOW",
                    liquidity="MEDIUM",
                    allocation_percentage=30
                ),
                Investment(
                    type="SAVINGS",
                    amount=goal.target_amount * 0.1,
                    expected_return=0.02,
                    risk_level="LOW",
                    liquidity="HIGH",
                    allocation_percentage=10
                )
            ],
            monthly_contribution=(goal.target_amount - goal.current_amount) / 12,
            expected_timeline=12,
            risk_assessment="MODERATE",
            contingency_plans=[
                "Increase contributions if falling behind",
                "Adjust allocation if market conditions change",
                "Extended timeline option available"
            ]
        )

class StrategyOptimizer:
    """Optimizes and adjusts strategies based on performance"""
    
    @staticmethod
    def optimize_strategy(
        strategy: Strategy,
        goal: FinancialGoal,
        market_conditions: Dict[str, str]
    ) -> Strategy:
        messages = [
            HumanMessage(content=f"""
            Optimize this investment strategy based on current conditions:
            
            Current Strategy:
            {json.dumps(asdict(strategy), indent=2)}
            
            Goal Progress:
            {json.dumps(asdict(goal), indent=2)}
            
            Market Conditions:
            {json.dumps(market_conditions, indent=2)}
            
            Please recommend:
            1. Allocation adjustments
            2. Contribution modifications
            3. Risk management updates
            4. Timeline revisions
            
            Explain the reasoning for each recommendation.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Update strategy based on recommendations
        # (Simplified - would parse LLM response in real implementation)
        if goal.progress < 50:  # If behind schedule
            strategy.monthly_contribution *= 1.2  # Increase contributions by 20%
            # Adjust allocations for more aggressive growth
            for inv in strategy.investments:
                if inv.type == "STOCKS":
                    inv.allocation_percentage += 10
                elif inv.type == "SAVINGS":
                    inv.allocation_percentage -= 10
        
        return strategy

def analyze_goals(state: PlannerState) -> PlannerState:
    """
    Analyze progress towards financial goals
    """
    analysis_results = {}
    
    for goal_id, goal_data in state["financial_goals"].items():
        goal = FinancialGoal(**goal_data)
        strategy = Strategy(**state["strategies"][goal_id]) if goal_id in state["strategies"] else None
        
        if strategy:
            analyzer = GoalAnalyzer()
            analysis_results[goal_id] = analyzer.analyze_goal_progress(goal, strategy)
    
    state["analysis_results"] = analysis_results
    state["next_action"] = "PLAN"
    
    return state

def plan_strategies(state: PlannerState) -> PlannerState:
    """
    Create or update investment strategies
    """
    planner = StrategyPlanner()
    optimizer = StrategyOptimizer()
    
    for goal_id, goal_data in state["financial_goals"].items():
        goal = FinancialGoal(**goal_data)
        
        if goal_id not in state["strategies"]:
            # Create new strategy
            strategy = planner.create_strategy(
                goal,
                state["current_portfolio"],
                state["market_conditions"]
            )
            state["strategies"][goal_id] = asdict(strategy)
        elif state["analysis_results"][goal_id]["requires_strategy_update"]:
            # Optimize existing strategy
            current_strategy = Strategy(**state["strategies"][goal_id])
            optimized_strategy = optimizer.optimize_strategy(
                current_strategy,
                goal,
                state["market_conditions"]
            )
            state["strategies"][goal_id] = asdict(optimized_strategy)
    
    state["next_action"] = "END"
    return state

def router(state: PlannerState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(PlannerState)

# Add nodes
workflow.add_node("analyze", analyze_goals)
workflow.add_node("plan", plan_strategies)

# Add edges
workflow.add_edge("analyze", router)
workflow.add_edge("plan", router)

# Set entry point
workflow.set_entry_point("analyze")

# Create conditional edges
workflow.add_conditional_edges(
    "analyze",
    router,
    {
        "PLAN": "plan",
        "END": END
    }
)

workflow.add_conditional_edges(
    "plan",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "financial_goals": {
            "RETIREMENT": {
                "id": "RETIREMENT",
                "name": "Retirement Fund",
                "target_amount": 2000000.0,
                "current_amount": 500000.0,
                "target_date": "2045-01-01",
                "priority": 1,
                "risk_tolerance": "MEDIUM",
                "progress": 25.0
            }
        },
        "current_portfolio": {
            "STOCKS": 300000.0,
            "BONDS": 150000.0,
            "SAVINGS": 50000.0
        },
        "market_conditions": {
            "stocks_outlook": "POSITIVE",
            "interest_rates": "RISING",
            "economic_indicators": "STABLE",
            "market_volatility": "MODERATE"
        },
        "strategies": {},
        "analysis_results": None,
        "next_action": "ANALYZE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 10. ระบบจดจำบริบท (Contextual Memory)

รูปแบบนี้ช่วยให้ระบบสามารถจดจำและใช้ประโยชน์จากข้อมูลในอดีต โดย:

- เก็บข้อมูลการโต้ตอบกับผู้ใช้
- วิเคราะห์รูปแบบการใช้งาน
- ปรับการทำงานให้เหมาะกับแต่ละผู้ใช้

ระบบแชทบอทที่จำความชอบของผู้ใช้และปรับการสนทนาให้เหมาะสมเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Contextual Memory***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from collections import defaultdict
import os

# Define data structures
@dataclass
class UserPreference:
    topic: str
    sentiment: float  # -1 to 1
    frequency: int
    last_discussed: str
    keywords: List[str]

@dataclass
class ConversationStyle:
    formality_level: str  # CASUAL, FORMAL, PROFESSIONAL
    preferred_language: str
    communication_pace: str  # BRIEF, DETAILED
    interests: List[str]
    special_notes: List[str]

@dataclass
class Interaction:
    timestamp: str
    user_message: str
    bot_response: str
    topics: List[str]
    sentiment: float
    user_feedback: Optional[str]

class ChatbotState(TypedDict):
    messages: List[str]
    current_input: str
    user_id: str
    user_preferences: Dict[str, Dict]
    conversation_style: Optional[Dict]
    interaction_history: List[Dict]
    context_analysis: Optional[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.7,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Contextual Chatbot"
    }
)

class PreferenceAnalyzer:
    """Analyzes and maintains user preferences"""
    
    @staticmethod
    def update_preferences(
        current_prefs: Dict[str, UserPreference],
        interaction: Interaction
    ) -> Dict[str, UserPreference]:
        messages = [
            HumanMessage(content=f"""
            Analyze this interaction and update user preferences:
            
            User Message: {interaction.user_message}
            Bot Response: {interaction.bot_response}
            Current Topics: {interaction.topics}
            
            Current Preferences:
            {json.dumps({k: asdict(v) for k, v in current_prefs.items()}, indent=2)}
            
            Please identify:
            1. New topics/interests
            2. Sentiment towards topics
            3. Key preferences or dislikes
            4. Patterns in communication style
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Update preferences based on analysis
        updated_prefs = current_prefs.copy()
        
        # Add new topics or update existing ones
        for topic in interaction.topics:
            if topic not in updated_prefs:
                updated_prefs[topic] = UserPreference(
                    topic=topic,
                    sentiment=interaction.sentiment,
                    frequency=1,
                    last_discussed=interaction.timestamp,
                    keywords=[]
                )
            else:
                pref = updated_prefs[topic]
                pref.frequency += 1
                pref.last_discussed = interaction.timestamp
                pref.sentiment = (pref.sentiment * (pref.frequency - 1) + interaction.sentiment) / pref.frequency
        
        return updated_prefs

class ConversationAnalyzer:
    """Analyzes conversation patterns and style"""
    
    @staticmethod
    def analyze_style(interactions: List[Interaction]) -> ConversationStyle:
        messages = [
            HumanMessage(content=f"""
            Analyze these interactions to determine conversation style:
            
            Recent Interactions:
            {json.dumps([asdict(i) for i in interactions[-5:]], indent=2)}
            
            Please determine:
            1. Preferred formality level
            2. Communication style (brief/detailed)
            3. Key interests and themes
            4. Special considerations
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create conversation style (simplified - would parse LLM response in real implementation)
        return ConversationStyle(
            formality_level="CASUAL" if "casual" in response.content.lower() else "FORMAL",
            preferred_language="English",  # Would be detected from interactions
            communication_pace="BRIEF" if "brief" in response.content.lower() else "DETAILED",
            interests=[topic for i in interactions[-5:] for topic in i.topics],
            special_notes=[]
        )

class ContextManager:
    """Manages contextual memory and retrieval"""
    
    def __init__(self):
        self.short_term_memory: List[Interaction] = []
        self.long_term_memory: Dict[str, List[Interaction]] = defaultdict(list)
    
    def add_interaction(self, interaction: Interaction, topics: List[str]):
        # Add to short-term memory
        self.short_term_memory.append(interaction)
        if len(self.short_term_memory) > 10:  # Keep last 10 interactions
            self.short_term_memory.pop(0)
        
        # Add to long-term memory by topic
        for topic in topics:
            self.long_term_memory[topic].append(interaction)
    
    def get_relevant_context(self, current_input: str, user_preferences: Dict[str, UserPreference]) -> List[Interaction]:
        messages = [
            HumanMessage(content=f"""
            Find relevant context for this input:
            
            Current Input: {current_input}
            User Preferences: {json.dumps({k: asdict(v) for k, v in user_preferences.items()}, indent=2)}
            
            Return relevant topics and importance scores.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Get relevant interactions (simplified)
        relevant = self.short_term_memory[-3:]  # Last 3 interactions
        for topic in user_preferences:
            if topic.lower() in current_input.lower():
                relevant.extend(self.long_term_memory[topic][-2:])  # Last 2 topic-specific interactions
        
        return list(set(relevant))  # Remove duplicates

class ResponseGenerator:
    """Generates contextually aware responses"""
    
    @staticmethod
    def generate_response(
        current_input: str,
        relevant_context: List[Interaction],
        conversation_style: ConversationStyle,
        user_preferences: Dict[str, UserPreference]
    ) -> str:
        context_str = "\n".join([
            f"Previous interaction: {i.user_message} -> {i.bot_response}"
            for i in relevant_context[-3:]  # Last 3 relevant interactions
        ])
        
        messages = [
            HumanMessage(content=f"""
            Generate a response considering this context:
            
            Current Input: {current_input}
            
            Previous Context:
            {context_str}
            
            Conversation Style:
            {json.dumps(asdict(conversation_style), indent=2)}
            
            User Preferences:
            {json.dumps({k: asdict(v) for k, v in user_preferences.items()}, indent=2)}
            
            Generate a response that:
            1. Matches the user's preferred style
            2. References relevant past interactions
            3. Shows awareness of user preferences
            4. Maintains conversation continuity
            """)
        ]
        
        response = llm.invoke(messages)
        return response.content

def analyze_context(state: ChatbotState) -> ChatbotState:
    """
    Analyze context and user preferences
    """
    if state["interaction_history"]:
        # Convert recent interactions to objects
        recent_interactions = [
            Interaction(**interaction)
            for interaction in state["interaction_history"][-5:]
        ]
        
        # Analyze conversation style
        analyzer = ConversationAnalyzer()
        conversation_style = analyzer.analyze_style(recent_interactions)
        state["conversation_style"] = asdict(conversation_style)
        
        # Get relevant context
        context_manager = ContextManager()
        for interaction in state["interaction_history"]:
            context_manager.add_interaction(
                Interaction(**interaction),
                interaction.get("topics", [])
            )
        
        current_prefs = {
            k: UserPreference(**v)
            for k, v in state["user_preferences"].items()
        }
        
        relevant_context = context_manager.get_relevant_context(
            state["current_input"],
            current_prefs
        )
        
        state["context_analysis"] = {
            "relevant_interactions": [asdict(i) for i in relevant_context],
            "conversation_style": asdict(conversation_style)
        }
    
    state["next_action"] = "RESPOND"
    return state

def generate_response(state: ChatbotState) -> ChatbotState:
    """
    Generate contextually aware response
    """
    # Convert data structures
    conversation_style = ConversationStyle(**state["conversation_style"])
    user_preferences = {
        k: UserPreference(**v)
        for k, v in state["user_preferences"].items()
    }
    relevant_context = [
        Interaction(**i)
        for i in state["context_analysis"]["relevant_interactions"]
    ]
    
    # Generate response
    generator = ResponseGenerator()
    response = generator.generate_response(
        state["current_input"],
        relevant_context,
        conversation_style,
        user_preferences
    )
    
    # Create new interaction
    interaction = Interaction(
        timestamp=datetime.now().isoformat(),
        user_message=state["current_input"],
        bot_response=response,
        topics=[],  # Would be extracted from response
        sentiment=0.0,  # Would be analyzed
        user_feedback=None
    )
    
    # Update state
    state["messages"].append(response)
    state["interaction_history"].append(asdict(interaction))
    state["next_action"] = "END"
    
    return state

def router(state: ChatbotState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(ChatbotState)

# Add nodes
workflow.add_node("analyze", analyze_context)
workflow.add_node("respond", generate_response)

# Add edges
workflow.add_edge("analyze", router)
workflow.add_edge("respond", router)

# Set entry point
workflow.set_entry_point("analyze")

# Create conditional edges
workflow.add_conditional_edges(
    "analyze",
    router,
    {
        "RESPOND": "respond",
        "END": END
    }
)

workflow.add_conditional_edges(
    "respond",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "current_input": "What restaurants do you recommend?",
        "user_id": "user123",
        "user_preferences": {
            "cuisine": {
                "topic": "cuisine",
                "sentiment": 0.8,
                "frequency": 5,
                "last_discussed": "2024-02-05T10:00:00",
                "keywords": ["Thai", "Italian", "vegetarian"]
            }
        },
        "conversation_style": {
            "formality_level": "CASUAL",
            "preferred_language": "English",
            "communication_pace": "DETAILED",
            "interests": ["food", "travel"],
            "special_notes": ["Prefers detailed explanations"]
        },
        "interaction_history": [
            {
                "timestamp": "2024-02-05T09:00:00",
                "user_message": "I love Thai food",
                "bot_response": "Thai cuisine is amazing! Do you have a favorite dish?",
                "topics": ["cuisine", "Thai"],
                "sentiment": 0.9,
                "user_feedback": None
            }
        ],
        "context_analysis": None,
        "next_action": "ANALYZE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}


### 11. ระบบหลายตัวแทนที่ทำงานร่วมกัน (Collaborative Multi-Agent Systems)

รูปแบบนี้น่าสนใจเพราะช่วยให้ระบบย่อยหลายๆ ระบบทำงานร่วมกันได้อย่างมีประสิทธิภาพ โดย:

- แบ่งงานตามความเชี่ยวชาญ
- ประสานงานระหว่างระบบย่อย
- แก้ไขความขัดแย้งที่อาจเกิดขึ้น

โดรนขนส่งที่ทำงานประสานกันเพื่อส่งพัสดุในเมืองเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Collaborative Multi-Agent Systems***" >}}
```python
from typing import Dict, List, TypedDict, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
import os
import random
from enum import Enum

# Define data structures
class DroneStatus(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    DELIVERING = "DELIVERING"
    RETURNING = "RETURNING"
    CHARGING = "CHARGING"
    MAINTENANCE = "MAINTENANCE"

@dataclass
class Location:
    x: float
    y: float
    name: str
    is_charging_station: bool = False
    is_depot: bool = False

@dataclass
class Package:
    id: str
    pickup_location: Location
    delivery_location: Location
    weight: float
    priority: int  # 1 (highest) to 5 (lowest)
    deadline: str
    status: str  # PENDING, ASSIGNED, IN_TRANSIT, DELIVERED

@dataclass
class Drone:
    id: str
    current_location: Location
    status: DroneStatus
    battery_level: float
    max_payload: float
    current_package: Optional[str]
    assigned_zone: List[float]  # [x_min, y_min, x_max, y_max]
    delivery_history: List[str]

@dataclass
class DeliveryPlan:
    drone_id: str
    package_ids: List[str]
    route: List[Location]
    estimated_battery_usage: float
    estimated_completion_time: str
    priority_score: float

class SystemState(TypedDict):
    messages: List[str]
    drones: Dict[str, Dict]
    packages: Dict[str, Dict]
    delivery_plans: Dict[str, Dict]
    conflict_resolutions: List[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Drone Fleet Manager"
    }
)

class RouteOptimizer:
    """Optimizes delivery routes for each drone"""
    
    @staticmethod
    def calculate_distance(loc1: Location, loc2: Location) -> float:
        return ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5
    
    @staticmethod
    def estimate_battery_usage(route: List[Location], package_weight: float) -> float:
        total_distance = sum(
            RouteOptimizer.calculate_distance(route[i], route[i+1])
            for i in range(len(route)-1)
        )
        # Simple battery usage model: distance + weight factor
        return total_distance * (1 + package_weight * 0.1)
    
    @staticmethod
    def optimize_route(
        drone: Drone,
        packages: List[Package],
        charging_stations: List[Location]
    ) -> DeliveryPlan:
        messages = [
            HumanMessage(content=f"""
            Optimize delivery route for drone considering:
            
            Drone Status:
            {json.dumps(asdict(drone), indent=2)}
            
            Available Packages:
            {json.dumps([asdict(p) for p in packages], indent=2)}
            
            Charging Stations:
            {json.dumps([asdict(s) for s in charging_stations], indent=2)}
            
            Consider:
            1. Battery efficiency
            2. Package priorities
            3. Deadlines
            4. Drone zone restrictions
            5. Charging station locations
            
            Provide:
            1. Optimized package sequence
            2. Complete route with charging stops
            3. Estimated battery usage
            4. Expected completion time
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create delivery plan (simplified - would parse LLM response in real implementation)
        route = [drone.current_location]
        if packages:
            route.extend([
                packages[0].pickup_location,
                packages[0].delivery_location
            ])
        if drone.battery_level < 0.3:
            route.append(min(
                charging_stations,
                key=lambda s: RouteOptimizer.calculate_distance(route[-1], s)
            ))
        
        return DeliveryPlan(
            drone_id=drone.id,
            package_ids=[p.id for p in packages],
            route=route,
            estimated_battery_usage=RouteOptimizer.estimate_battery_usage(
                route,
                sum(p.weight for p in packages)
            ),
            estimated_completion_time=(
                datetime.now() + timedelta(minutes=len(route)*10)
            ).isoformat(),
            priority_score=sum(1/p.priority for p in packages)
        )

class ConflictResolver:
    """Resolves conflicts between drone delivery plans"""
    
    @staticmethod
    def detect_conflicts(plans: List[DeliveryPlan]) -> List[Dict]:
        messages = [
            HumanMessage(content=f"""
            Analyze these delivery plans for potential conflicts:
            
            Delivery Plans:
            {json.dumps([asdict(p) for p in plans], indent=2)}
            
            Check for:
            1. Path intersections
            2. Resource conflicts (charging stations)
            3. Timing conflicts
            4. Zone violations
            
            Identify specific conflicts and suggest resolutions.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Process conflicts (simplified)
        conflicts = []
        for i, plan1 in enumerate(plans):
            for plan2 in plans[i+1:]:
                # Check for path intersections
                for loc1 in plan1.route:
                    for loc2 in plan2.route:
                        if (loc1.x == loc2.x and loc1.y == loc2.y and
                            loc1.is_charging_station):
                            conflicts.append({
                                "type": "CHARGING_STATION_CONFLICT",
                                "drones": [plan1.drone_id, plan2.drone_id],
                                "location": asdict(loc1),
                                "resolution": "RESEQUENCE"
                            })
        
        return conflicts
    
    @staticmethod
    def resolve_conflicts(
        conflicts: List[Dict],
        plans: Dict[str, DeliveryPlan]
    ) -> Dict[str, DeliveryPlan]:
        messages = [
            HumanMessage(content=f"""
            Resolve these delivery plan conflicts:
            
            Conflicts:
            {json.dumps(conflicts, indent=2)}
            
            Current Plans:
            {json.dumps({k: asdict(v) for k, v in plans.items()}, indent=2)}
            
            Provide:
            1. Modified routes
            2. Updated timing
            3. Rationale for changes
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Resolve conflicts (simplified)
        updated_plans = plans.copy()
        for conflict in conflicts:
            if conflict["type"] == "CHARGING_STATION_CONFLICT":
                # Add delay to second drone's plan
                drone_id = conflict["drones"][1]
                plan = updated_plans[drone_id]
                plan.estimated_completion_time = (
                    datetime.fromisoformat(plan.estimated_completion_time) +
                    timedelta(minutes=15)
                ).isoformat()
        
        return updated_plans

class FleetCoordinator:
    """Coordinates overall fleet operations"""
    
    @staticmethod
    def assign_packages(
        drones: Dict[str, Drone],
        packages: List[Package],
        charging_stations: List[Location]
    ) -> Dict[str, DeliveryPlan]:
        messages = [
            HumanMessage(content=f"""
            Assign packages to drones optimally:
            
            Available Drones:
            {json.dumps({k: asdict(v) for k, v in drones.items()}, indent=2)}
            
            Pending Packages:
            {json.dumps([asdict(p) for p in packages], indent=2)}
            
            Consider:
            1. Drone locations and zones
            2. Package priorities and deadlines
            3. Battery levels
            4. Load balancing
            
            Provide assignment plan with rationale.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create delivery plans (simplified)
        optimizer = RouteOptimizer()
        plans = {}
        
        for drone in drones.values():
            if drone.status in [DroneStatus.IDLE, DroneStatus.RETURNING]:
                # Find packages in drone's zone
                zone_packages = [
                    p for p in packages
                    if (drone.assigned_zone[0] <= p.delivery_location.x <= drone.assigned_zone[2] and
                        drone.assigned_zone[1] <= p.delivery_location.y <= drone.assigned_zone[3])
                ]
                if zone_packages:
                    plans[drone.id] = optimizer.optimize_route(
                        drone,
                        sorted(zone_packages, key=lambda p: p.priority)[:2],
                        charging_stations
                    )
        
        return plans

def plan_deliveries(state: SystemState) -> SystemState:
    """
    Plan and optimize delivery routes
    """
    # Convert data structures
    drones = {
        k: Drone(**d)
        for k, d in state["drones"].items()
    }
    packages = [
        Package(**p)
        for p in state["packages"].values()
        if p["status"] == "PENDING"
    ]
    charging_stations = [
        Location(0, 0, "Station 1", True),
        Location(10, 10, "Station 2", True)
    ]
    
    # Create delivery plans
    coordinator = FleetCoordinator()
    plans = coordinator.assign_packages(drones, packages, charging_stations)
    
    # Convert to dict format
    state["delivery_plans"] = {
        k: asdict(v)
        for k, v in plans.items()
    }
    state["next_action"] = "RESOLVE_CONFLICTS"
    
    return state

def resolve_conflicts(state: SystemState) -> SystemState:
    """
    Detect and resolve conflicts between delivery plans
    """
    if not state["delivery_plans"]:
        state["next_action"] = "END"
        return state
    
    # Convert plans to objects
    plans = {
        k: DeliveryPlan(**p)
        for k, p in state["delivery_plans"].items()
    }
    
    # Detect and resolve conflicts
    resolver = ConflictResolver()
    conflicts = resolver.detect_conflicts(list(plans.values()))
    
    if conflicts:
        updated_plans = resolver.resolve_conflicts(conflicts, plans)
        state["delivery_plans"] = {
            k: asdict(v)
            for k, v in updated_plans.items()
        }
        state["conflict_resolutions"] = conflicts
    
    state["next_action"] = "END"
    return state

def router(state: SystemState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(SystemState)

# Add nodes
workflow.add_node("plan", plan_deliveries)
workflow.add_node("resolve", resolve_conflicts)

# Add edges
workflow.add_edge("plan", router)
workflow.add_edge("resolve", router)

# Set entry point
workflow.set_entry_point("plan")

# Create conditional edges
workflow.add_conditional_edges(
    "plan",
    router,
    {
        "RESOLVE_CONFLICTS": "resolve",
        "END": END
    }
)

workflow.add_conditional_edges(
    "resolve",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "drones": {
            "drone1": {
                "id": "drone1",
                "current_location": asdict(Location(0, 0, "Depot", is_depot=True)),
                "status": "IDLE",
                "battery_level": 0.9,
                "max_payload": 5.0,
                "current_package": None,
                "assigned_zone": [0, 0, 5, 5],
                "delivery_history": []
            },
            "drone2": {
                "id": "drone2",
                "current_location": asdict(Location(5, 5, "Depot 2", is_depot=True)),
                "status": "IDLE",
                "battery_level": 0.8,
                "max_payload": 5.0,
                "current_package": None,
                "assigned_zone": [5, 5, 10, 10],
                "delivery_history": []
            }
        },
        "packages": {
            "pkg1": {
                "id": "pkg1",
                "pickup_location": asdict(Location(1, 1, "Warehouse A")),
                "delivery_location": asdict(Location(4, 4, "Customer 1")),
                "weight": 2.0,
                "priority": 1,
                "deadline": (datetime.now() + timedelta(hours=2)).isoformat(),
                "status": "PENDING"
            },
            "pkg2": {
                "id": "pkg2",
                "pickup_location": asdict(Location(6, 6, "Warehouse B")),
                "delivery_location": asdict(Location(8, 8, "Customer 2")),
                "weight": 3.0,
                "priority": 2,
                "deadline": (datetime.now() + timedelta(hours=3)).isoformat(),
                "status": "PENDING"
            }
        },
        "delivery_plans": {},
        "conflict_resolutions": [],
        "next_action": "PLAN"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 12. ระบบสำรวจ (Exploratory Agent)

รูปแบบนี้เหมาะกับการค้นหาข้อมูลและโอกาสใหม่ๆ โดย:

- สำรวจสภาพแวดล้อมหรือข้อมูลที่ไม่คุ้นเคย
- วิเคราะห์และจัดเก็บข้อมูลที่พบ
- ระบุรูปแบบหรือโอกาสที่น่าสนใจ

ผู้ช่วยวิจัยที่สแกนวารสารวิชาการเพื่อค้นหาแนวโน้มใหม่ๆ เป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Exploratory Agent***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from collections import defaultdict
import os

# Define data structures
@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    publication_date: str
    journal: str
    citations: int
    research_areas: List[str]

@dataclass
class ResearchTrend:
    topic: str
    emerging_keywords: List[str]
    key_papers: List[str]
    growth_rate: float  # Trend growth rate
    relevance_score: float  # 0-1
    first_observed: str
    last_updated: str

@dataclass
class Insight:
    trend_id: str
    description: str
    supporting_evidence: List[str]
    potential_impact: str
    confidence_score: float
    timestamp: str

class ResearchState(TypedDict):
    messages: List[str]
    papers: Dict[str, Dict]
    identified_trends: Dict[str, Dict]
    insights: List[Dict]
    research_focus: List[str]
    analysis_results: Optional[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.3,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Research Assistant"
    }
)

class TrendAnalyzer:
    """Analyzes research papers to identify trends"""
    
    @staticmethod
    def identify_trends(papers: List[ResearchPaper]) -> List[ResearchTrend]:
        # Group papers by research areas
        area_papers = defaultdict(list)
        for paper in papers:
            for area in paper.research_areas:
                area_papers[area].append(paper)
        
        messages = [
            HumanMessage(content=f"""
            Analyze these research papers and identify emerging trends:
            
            Papers by Research Area:
            {json.dumps({area: [asdict(p) for p in papers] for area, papers in area_papers.items()}, indent=2)}
            
            For each trend, identify:
            1. Core topic and theme
            2. Key emerging keywords
            3. Most influential papers
            4. Growth trajectory
            5. Potential impact
            
            Focus on:
            - Novel research directions
            - Emerging methodologies
            - Cross-disciplinary connections
            - Technology applications
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Process trends (simplified - would parse LLM response in real implementation)
        trends = []
        for area, area_paper_list in area_papers.items():
            if len(area_paper_list) >= 3:  # Minimum papers to identify trend
                recent_papers = sorted(
                    area_paper_list,
                    key=lambda p: p.publication_date,
                    reverse=True
                )[:3]
                
                trend = ResearchTrend(
                    topic=area,
                    emerging_keywords=list(set(
                        kw for p in recent_papers for kw in p.keywords
                    )),
                    key_papers=[p.title for p in recent_papers],
                    growth_rate=0.5,  # Would calculate from citation patterns
                    relevance_score=0.8,  # Would calculate based on analysis
                    first_observed=min(p.publication_date for p in area_paper_list),
                    last_updated=max(p.publication_date for p in area_paper_list)
                )
                trends.append(trend)
        
        return trends

class InsightGenerator:
    """Generates insights from identified trends"""
    
    @staticmethod
    def generate_insights(
        trends: List[ResearchTrend],
        papers: Dict[str, ResearchPaper]
    ) -> List[Insight]:
        messages = [
            HumanMessage(content=f"""
            Generate research insights based on these trends:
            
            Research Trends:
            {json.dumps([asdict(t) for t in trends], indent=2)}
            
            Supporting Papers:
            {json.dumps({k: asdict(v) for k, v in papers.items()}, indent=2)}
            
            For each insight:
            1. Describe the key finding
            2. Provide supporting evidence
            3. Assess potential impact
            4. Estimate confidence level
            
            Consider:
            - Cross-trend patterns
            - Unexpected connections
            - Research gaps
            - Future implications
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Generate insights (simplified)
        insights = []
        for trend in trends:
            insight = Insight(
                trend_id=trend.topic,
                description=f"Emerging trend in {trend.topic}",
                supporting_evidence=trend.key_papers,
                potential_impact="HIGH" if trend.growth_rate > 0.7 else "MEDIUM",
                confidence_score=trend.relevance_score,
                timestamp=datetime.now().isoformat()
            )
            insights.append(insight)
        
        return insights

class PatternMatcher:
    """Identifies patterns and connections across research areas"""
    
    @staticmethod
    def find_patterns(
        trends: List[ResearchTrend],
        insights: List[Insight]
    ) -> Dict:
        messages = [
            HumanMessage(content=f"""
            Identify patterns and connections across research trends:
            
            Trends:
            {json.dumps([asdict(t) for t in trends], indent=2)}
            
            Insights:
            {json.dumps([asdict(i) for i in insights], indent=2)}
            
            Look for:
            1. Common themes across areas
            2. Complementary research directions
            3. Technology convergence
            4. Methodology patterns
            
            Highlight:
            - Strong connections
            - Research opportunities
            - Potential collaborations
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Analyze patterns (simplified)
        patterns = {
            "theme_clusters": defaultdict(list),
            "methodology_patterns": [],
            "research_opportunities": []
        }
        
        # Group related trends
        for trend in trends:
            for keyword in trend.emerging_keywords:
                patterns["theme_clusters"][keyword].append(trend.topic)
        
        return patterns

def analyze_trends(state: ResearchState) -> ResearchState:
    """
    Identify and analyze research trends
    """
    # Convert papers to objects
    papers = [
        ResearchPaper(**paper_data)
        for paper_data in state["papers"].values()
    ]
    
    # Identify trends
    analyzer = TrendAnalyzer()
    trends = analyzer.identify_trends(papers)
    
    # Store trends
    state["identified_trends"] = {
        trend.topic: asdict(trend)
        for trend in trends
    }
    
    state["next_action"] = "GENERATE_INSIGHTS"
    return state

def generate_insights(state: ResearchState) -> ResearchState:
    """
    Generate insights from identified trends
    """
    # Convert data structures
    trends = [
        ResearchTrend(**trend_data)
        for trend_data in state["identified_trends"].values()
    ]
    papers = {
        k: ResearchPaper(**v)
        for k, v in state["papers"].items()
    }
    
    # Generate insights
    generator = InsightGenerator()
    insights = generator.generate_insights(trends, papers)
    
    # Find patterns
    matcher = PatternMatcher()
    patterns = matcher.find_patterns(trends, insights)
    
    # Update state
    state["insights"] = [asdict(insight) for insight in insights]
    state["analysis_results"] = patterns
    state["next_action"] = "END"
    
    return state

def router(state: ResearchState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("analyze", analyze_trends)
workflow.add_node("insights", generate_insights)

# Add edges
workflow.add_edge("analyze", router)
workflow.add_edge("insights", router)

# Set entry point
workflow.set_entry_point("analyze")

# Create conditional edges
workflow.add_conditional_edges(
    "analyze",
    router,
    {
        "GENERATE_INSIGHTS": "insights",
        "END": END
    }
)

workflow.add_conditional_edges(
    "insights",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state with example papers
    initial_state = {
        "messages": [],
        "papers": {
            "paper1": {
                "title": "Deep Learning in Medical Imaging",
                "authors": ["Smith, J.", "Jones, K."],
                "abstract": "This paper explores applications of deep learning...",
                "keywords": ["deep learning", "medical imaging", "AI"],
                "publication_date": "2024-01-15",
                "journal": "AI in Medicine",
                "citations": 10,
                "research_areas": ["AI", "Healthcare"]
            },
            "paper2": {
                "title": "Advances in Quantum Computing",
                "authors": ["Brown, R.", "Lee, M."],
                "abstract": "Recent developments in quantum computing...",
                "keywords": ["quantum computing", "qubits", "algorithms"],
                "publication_date": "2024-01-20",
                "journal": "Quantum Computing Review",
                "citations": 15,
                "research_areas": ["Quantum Computing", "Computer Science"]
            }
        },
        "identified_trends": {},
        "insights": [],
        "research_focus": ["AI", "Quantum Computing", "Healthcare"],
        "analysis_results": None,
        "next_action": "ANALYZE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 13. ระบบจัดการขั้นตอนการทำงานแบบปรับตัวได้ (Adaptive Workflow Orchestration)

รูปแบบนี้ช่วยให้ระบบปรับเปลี่ยนการทำงานตามสถานการณ์ได้อย่างยืดหยุ่น โดย:

- ติดตามการเปลี่ยนแปลงของสภาพแวดล้อม
- ปรับลำดับความสำคัญของงาน
- จัดสรรทรัพยากรใหม่ตามความจำเป็น

ระบบบริหารจัดการโรงพยาบาลที่ปรับการจัดสรรทรัพยากรตามจำนวนผู้ป่วยที่เข้ามาเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Adaptive Workflow Orchestration***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from enum import Enum
import os

# Define data structures
class ResourceType(str, Enum):
    DOCTOR = "DOCTOR"
    NURSE = "NURSE"
    BED = "BED"
    EQUIPMENT = "EQUIPMENT"
    ICU = "ICU"
    EMERGENCY = "EMERGENCY"

class PatientPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Resource:
    id: str
    type: ResourceType
    department: str
    status: str  # AVAILABLE, BUSY, MAINTENANCE
    current_assignment: Optional[str]
    capacity: float  # 0-1 for utilization
    skills: List[str]

@dataclass
class Department:
    name: str
    current_load: float  # 0-1 for utilization
    patient_count: int
    resources: Dict[ResourceType, List[str]]
    wait_time: int  # minutes
    status: str  # NORMAL, HIGH_LOAD, CRITICAL

@dataclass
class Patient:
    id: str
    priority: PatientPriority
    department: str
    required_resources: List[ResourceType]
    arrival_time: str
    status: str  # WAITING, IN_TREATMENT, DISCHARGED
    estimated_duration: int  # minutes

@dataclass
class ResourceAllocation:
    patient_id: str
    resource_ids: List[str]
    department: str
    start_time: str
    duration: int
    priority: PatientPriority

class HospitalState(TypedDict):
    messages: List[str]
    departments: Dict[str, Dict]
    resources: Dict[str, Dict]
    patients: Dict[str, Dict]
    allocations: Dict[str, Dict]
    metrics: Dict[str, float]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Hospital Resource Manager"
    }
)

class LoadAnalyzer:
    """Analyzes department loads and resource utilization"""
    
    @staticmethod
    def analyze_department_loads(
        departments: Dict[str, Department],
        patients: Dict[str, Patient]
    ) -> Dict[str, float]:
        messages = [
            HumanMessage(content=f"""
            Analyze hospital department loads:
            
            Departments:
            {json.dumps({k: asdict(v) for k, v in departments.items()}, indent=2)}
            
            Current Patients:
            {json.dumps({k: asdict(v) for k, v in patients.items()}, indent=2)}
            
            Consider:
            1. Current patient count vs capacity
            2. Patient priorities and types
            3. Wait times
            4. Resource utilization
            
            Provide load analysis and recommendations.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Calculate loads (simplified)
        loads = {}
        for dept_name, dept in departments.items():
            dept_patients = sum(1 for p in patients.values() if p.department == dept_name)
            critical_patients = sum(
                1 for p in patients.values()
                if p.department == dept_name and p.priority == PatientPriority.CRITICAL
            )
            
            # Calculate load score (0-1)
            base_load = dept_patients / 20  # Assuming 20 patients is max capacity
            critical_factor = critical_patients * 0.1  # Extra 10% load per critical patient
            wait_time_factor = min(dept.wait_time / 120, 0.5)  # Max 50% impact from wait time
            
            loads[dept_name] = min(base_load + critical_factor + wait_time_factor, 1.0)
        
        return loads

class ResourceOptimizer:
    """Optimizes resource allocation based on loads and priorities"""
    
    @staticmethod
    def optimize_resources(
        departments: Dict[str, Department],
        resources: Dict[str, Resource],
        patients: Dict[str, Patient],
        current_loads: Dict[str, float]
    ) -> List[ResourceAllocation]:
        messages = [
            HumanMessage(content=f"""
            Optimize resource allocation based on current situation:
            
            Department Loads:
            {json.dumps(current_loads, indent=2)}
            
            Available Resources:
            {json.dumps({k: asdict(v) for k, v in resources.items()}, indent=2)}
            
            Waiting Patients:
            {json.dumps({k: asdict(v) for k, v in patients.items()
                        if v.status == "WAITING"}, indent=2)}
            
            Consider:
            1. Patient priorities
            2. Department loads
            3. Resource capabilities
            4. Wait times
            5. Resource utilization balance
            
            Provide optimal resource allocation plan.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create allocations (simplified)
        allocations = []
        waiting_patients = {k: p for k, p in patients.items() if p.status == "WAITING"}
        available_resources = {k: r for k, r in resources.items() if r.status == "AVAILABLE"}
        
        # Prioritize patients by severity and wait time
        sorted_patients = sorted(
            waiting_patients.values(),
            key=lambda p: (
                PatientPriority[p.priority].value,
                datetime.fromisoformat(p.arrival_time)
            )
        )
        
        for patient in sorted_patients:
            # Find available required resources
            needed_resources = []
            for resource_type in patient.required_resources:
                matching_resources = [
                    r for r in available_resources.values()
                    if r.type == resource_type and r.department == patient.department
                ]
                if matching_resources:
                    needed_resources.append(matching_resources[0].id)
                    del available_resources[matching_resources[0].id]
            
            if len(needed_resources) == len(patient.required_resources):
                allocation = ResourceAllocation(
                    patient_id=patient.id,
                    resource_ids=needed_resources,
                    department=patient.department,
                    start_time=datetime.now().isoformat(),
                    duration=patient.estimated_duration,
                    priority=patient.priority
                )
                allocations.append(allocation)
        
        return allocations

class WorkflowAdjuster:
    """Adjusts workflows based on current situation"""
    
    @staticmethod
    def adjust_workflows(
        departments: Dict[str, Department],
        loads: Dict[str, float],
        resources: Dict[str, Resource]
    ) -> Dict[str, List[str]]:
        messages = [
            HumanMessage(content=f"""
            Recommend workflow adjustments based on current situation:
            
            Department Status:
            {json.dumps({k: asdict(v) for k, v in departments.items()}, indent=2)}
            
            Department Loads:
            {json.dumps(loads, indent=2)}
            
            Available Resources:
            {json.dumps({k: asdict(v) for k, v in resources.items()}, indent=2)}
            
            Consider:
            1. Load balancing opportunities
            2. Resource reallocation needs
            3. Process optimization
            4. Emergency protocols
            
            Provide specific workflow adjustment recommendations.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Generate adjustments (simplified)
        adjustments = defaultdict(list)
        
        for dept_name, load in loads.items():
            if load > 0.8:  # High load
                adjustments[dept_name].extend([
                    "ACTIVATE_EMERGENCY_PROTOCOL",
                    "REQUEST_ADDITIONAL_STAFF",
                    "EXPEDITE_DISCHARGES"
                ])
            elif load > 0.6:  # Moderate load
                adjustments[dept_name].extend([
                    "OPTIMIZE_RESOURCE_ALLOCATION",
                    "REVIEW_WAIT_TIMES"
                ])
        
        return adjustments

def analyze_situation(state: HospitalState) -> HospitalState:
    """
    Analyze current hospital situation
    """
    # Convert data structures
    departments = {
        k: Department(**d)
        for k, d in state["departments"].items()
    }
    patients = {
        k: Patient(**p)
        for k, p in state["patients"].items()
    }
    
    # Analyze loads
    analyzer = LoadAnalyzer()
    loads = analyzer.analyze_department_loads(departments, patients)
    
    # Store analysis results
    state["metrics"]["department_loads"] = loads
    state["next_action"] = "OPTIMIZE"
    
    return state

def optimize_resources(state: HospitalState) -> HospitalState:
    """
    Optimize resource allocation
    """
    # Convert data structures
    departments = {
        k: Department(**d)
        for k, d in state["departments"].items()
    }
    resources = {
        k: Resource(**r)
        for k, r in state["resources"].items()
    }
    patients = {
        k: Patient(**p)
        for k, p in state["patients"].items()
    }
    
    # Optimize resources
    optimizer = ResourceOptimizer()
    allocations = optimizer.optimize_resources(
        departments,
        resources,
        patients,
        state["metrics"]["department_loads"]
    )
    
    # Update allocations
    state["allocations"] = {
        f"alloc_{i}": asdict(alloc)
        for i, alloc in enumerate(allocations)
    }
    
    state["next_action"] = "ADJUST"
    return state

def adjust_workflows(state: HospitalState) -> HospitalState:
    """
    Adjust workflows based on current situation
    """
    # Convert data structures
    departments = {
        k: Department(**d)
        for k, d in state["departments"].items()
    }
    resources = {
        k: Resource(**r)
        for k, r in state["resources"].items()
    }
    
    # Adjust workflows
    adjuster = WorkflowAdjuster()
    adjustments = adjuster.adjust_workflows(
        departments,
        state["metrics"]["department_loads"],
        resources
    )
    
    # Store adjustments
    state["metrics"]["workflow_adjustments"] = adjustments
    state["next_action"] = "END"
    
    return state

def router(state: HospitalState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(HospitalState)

# Add nodes
workflow.add_node("analyze", analyze_situation)
workflow.add_node("optimize", optimize_resources)
workflow.add_node("adjust", adjust_workflows)

# Add edges
workflow.add_edge("analyze", router)
workflow.add_edge("optimize", router)
workflow.add_edge("adjust", router)

# Set entry point
workflow.set_entry_point("analyze")

# Create conditional edges
workflow.add_conditional_edges(
    "analyze",
    router,
    {
        "OPTIMIZE": "optimize",
        "END": END
    }
)

workflow.add_conditional_edges(
    "optimize",
    router,
    {
        "ADJUST": "adjust",
        "END": END
    }
)

workflow.add_conditional_edges(
    "adjust",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "departments": {
            "emergency": {
                "name": "Emergency",
                "current_load": 0.7,
                "patient_count": 15,
                "resources": {
                    "DOCTOR": ["doc1", "doc2"],
                    "NURSE": ["nurse1", "nurse2"],
                    "BED": ["bed1", "bed2"]
                },
                "wait_time": 45,
                "status": "HIGH_LOAD"
            }
        },
        "resources": {
            "doc1": {
                "id": "doc1",
                "type": "DOCTOR",
                "department": "emergency",
                "status": "BUSY",
                "current_assignment": "patient1",
                "capacity": 0.8,
                "skills": ["emergency", "general"]
            }
        },
        "patients": {
            "patient1": {
                "id": "patient1",
                "priority": "HIGH",
                "department": "emergency",
                "required_resources": ["DOCTOR", "BED"],
                "arrival_time": "2024-02-06T10:00:00",
                "status": "IN_TREATMENT",
                "estimated_duration": 60
            }
        },
        "allocations": {},
        "metrics": {},
        "next_action": "ANALYZE"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 14. ระบบซ่อมแซมตัวเอง (Self-Healing Systems)

รูปแบบนี้น่าสนใจเพราะช่วยให้ระบบสามารถรักษาเสถียรภาพการทำงานได้ด้วยตัวเอง โดย:

- ตรวจจับปัญหาหรือข้อผิดพลาด
- วิเคราะห์สาเหตุของปัญหา
- ดำเนินการแก้ไขโดยอัตโนมัติ

ระบบจัดการคลาวด์ที่สามารถตรวจจับและแก้ไขปัญหาเซิร์ฟเวอร์ที่ทำงานผิดปกติเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Self-Healing Systems***" >}}
```python
from typing import Dict, List, TypedDict, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from enum import Enum
import os

# Define data structures
class ServerStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"

class IssueType(str, Enum):
    CPU_OVERLOAD = "CPU_OVERLOAD"
    MEMORY_LEAK = "MEMORY_LEAK"
    DISK_FULL = "DISK_FULL"
    NETWORK_LATENCY = "NETWORK_LATENCY"
    SERVICE_DOWN = "SERVICE_DOWN"
    DATABASE_SLOW = "DATABASE_SLOW"

@dataclass
class ServerMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    response_time: float
    error_rate: float
    uptime: float
    timestamp: str

@dataclass
class Server:
    id: str
    name: str
    status: ServerStatus
    role: str  # web, database, cache, etc.
    current_metrics: ServerMetrics
    historical_metrics: List[ServerMetrics]
    active_issues: List[str]
    maintenance_history: List[Dict]

@dataclass
class Issue:
    id: str
    server_id: str
    type: IssueType
    severity: float  # 0-1
    detected_time: str
    description: str
    root_cause: Optional[str]
    resolution_steps: List[str]
    status: str  # DETECTED, ANALYZING, RESOLVING, RESOLVED

@dataclass
class Resolution:
    issue_id: str
    server_id: str
    actions: List[str]
    expected_impact: Dict[str, float]
    success_criteria: Dict[str, float]
    rollback_plan: List[str]

class CloudState(TypedDict):
    messages: List[str]
    servers: Dict[str, Dict]
    issues: Dict[str, Dict]
    resolutions: Dict[str, Dict]
    system_health: Dict[str, float]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Cloud Health Monitor"
    }
)

class HealthMonitor:
    """Monitors server health and detects issues"""
    
    @staticmethod
    def analyze_metrics(server: Server) -> List[Issue]:
        messages = [
            HumanMessage(content=f"""
            Analyze these server metrics for potential issues:
            
            Server Info:
            {json.dumps(asdict(server), indent=2)}
            
            Key Metrics:
            - CPU Usage: {server.current_metrics.cpu_usage}%
            - Memory Usage: {server.current_metrics.memory_usage}%
            - Disk Usage: {server.current_metrics.disk_usage}%
            - Network Latency: {server.current_metrics.network_latency}ms
            - Error Rate: {server.current_metrics.error_rate}%
            
            Historical Metrics:
            {json.dumps([asdict(m) for m in server.historical_metrics[-5:]], indent=2)}
            
            Identify:
            1. Performance issues
            2. Resource constraints
            3. Service degradation
            4. Anomalous behavior
            
            For each issue provide:
            1. Issue type
            2. Severity
            3. Potential root causes
            4. Initial resolution steps
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Detect issues (simplified)
        issues = []
        metrics = server.current_metrics
        
        # Check CPU
        if metrics.cpu_usage > 90:
            issues.append(Issue(
                id=f"issue_{len(issues)}",
                server_id=server.id,
                type=IssueType.CPU_OVERLOAD,
                severity=min((metrics.cpu_usage - 90) / 10, 1.0),
                detected_time=datetime.now().isoformat(),
                description="CPU usage critically high",
                root_cause=None,
                resolution_steps=["Scale up CPU", "Check runaway processes"],
                status="DETECTED"
            ))
        
        # Check Memory
        if metrics.memory_usage > 85:
            issues.append(Issue(
                id=f"issue_{len(issues)}",
                server_id=server.id,
                type=IssueType.MEMORY_LEAK,
                severity=min((metrics.memory_usage - 85) / 15, 1.0),
                detected_time=datetime.now().isoformat(),
                description="Memory usage abnormally high",
                root_cause=None,
                resolution_steps=["Analyze memory dumps", "Check memory leaks"],
                status="DETECTED"
            ))
        
        return issues

class DiagnosticEngine:
    """Analyzes issues and determines root causes"""
    
    @staticmethod
    def analyze_issue(
        issue: Issue,
        server: Server,
        system_health: Dict[str, float]
    ) -> Issue:
        messages = [
            HumanMessage(content=f"""
            Analyze this server issue and determine root cause:
            
            Issue Details:
            {json.dumps(asdict(issue), indent=2)}
            
            Server Information:
            {json.dumps(asdict(server), indent=2)}
            
            System Health Context:
            {json.dumps(system_health, indent=2)}
            
            Provide:
            1. Detailed root cause analysis
            2. Impact assessment
            3. Recommended resolution steps
            4. Risk evaluation
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Update issue with analysis (simplified)
        if issue.type == IssueType.CPU_OVERLOAD:
            issue.root_cause = "High application load causing CPU spikes"
            issue.resolution_steps.extend([
                "Enable auto-scaling",
                "Optimize application code",
                "Add load balancing"
            ])
        elif issue.type == IssueType.MEMORY_LEAK:
            issue.root_cause = "Application memory leak detected"
            issue.resolution_steps.extend([
                "Restart problematic services",
                "Apply memory leak patches",
                "Monitor memory patterns"
            ])
        
        issue.status = "ANALYZING"
        return issue

class RepairExecutor:
    """Executes repair actions and monitors results"""
    
    @staticmethod
    def create_resolution_plan(
        issue: Issue,
        server: Server
    ) -> Resolution:
        messages = [
            HumanMessage(content=f"""
            Create a resolution plan for this issue:
            
            Issue:
            {json.dumps(asdict(issue), indent=2)}
            
            Server:
            {json.dumps(asdict(server), indent=2)}
            
            Provide:
            1. Detailed action steps
            2. Success criteria
            3. Expected impacts
            4. Rollback procedures
            5. Risk mitigation strategies
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Create resolution plan (simplified)
        return Resolution(
            issue_id=issue.id,
            server_id=server.id,
            actions=issue.resolution_steps,
            expected_impact={
                "cpu_usage": -20 if issue.type == IssueType.CPU_OVERLOAD else 0,
                "memory_usage": -30 if issue.type == IssueType.MEMORY_LEAK else 0,
                "response_time": -50
            },
            success_criteria={
                "cpu_usage": 70,
                "memory_usage": 70,
                "error_rate": 1
            },
            rollback_plan=[
                "Stop new resolution actions",
                "Restore from last known good configuration",
                "Reset affected services"
            ]
        )

def monitor_health(state: CloudState) -> CloudState:
    """
    Monitor server health and detect issues
    """
    # Convert data structures
    servers = {
        k: Server(**s)
        for k, s in state["servers"].items()
    }
    
    # Check each server
    monitor = HealthMonitor()
    new_issues = []
    
    for server in servers.values():
        server_issues = monitor.analyze_metrics(server)
        new_issues.extend(server_issues)
    
    # Update state
    for issue in new_issues:
        state["issues"][issue.id] = asdict(issue)
    
    # Calculate system health
    total_servers = len(servers)
    healthy_servers = sum(1 for s in servers.values() if s.status == ServerStatus.HEALTHY)
    state["system_health"] = {
        "overall_health": healthy_servers / total_servers,
        "total_issues": len(state["issues"]),
        "critical_issues": sum(1 for i in state["issues"].values() if float(i["severity"]) > 0.7)
    }
    
    state["next_action"] = "DIAGNOSE" if new_issues else "END"
    return state

def diagnose_issues(state: CloudState) -> CloudState:
    """
    Analyze issues and determine root causes
    """
    # Convert data structures
    issues = {
        k: Issue(**i)
        for k, i in state["issues"].items()
        if i["status"] == "DETECTED"
    }
    servers = {
        k: Server(**s)
        for k, s in state["servers"].items()
    }
    
    # Analyze each issue
    engine = DiagnosticEngine()
    for issue_id, issue in issues.items():
        server = servers[issue.server_id]
        updated_issue = engine.analyze_issue(
            issue,
            server,
            state["system_health"]
        )
        state["issues"][issue_id] = asdict(updated_issue)
    
    state["next_action"] = "REPAIR" if issues else "END"
    return state

def execute_repairs(state: CloudState) -> CloudState:
    """
    Execute repair actions for analyzed issues
    """
    # Convert data structures
    issues = {
        k: Issue(**i)
        for k, i in state["issues"].items()
        if i["status"] == "ANALYZING"
    }
    servers = {
        k: Server(**s)
        for k, s in state["servers"].items()
    }
    
    # Create and execute repair plans
    executor = RepairExecutor()
    for issue_id, issue in issues.items():
        server = servers[issue.server_id]
        resolution = executor.create_resolution_plan(issue, server)
        state["resolutions"][issue_id] = asdict(resolution)
        
        # Update issue status
        issue.status = "RESOLVING"
        state["issues"][issue_id] = asdict(issue)
    
    state["next_action"] = "END"
    return state

def router(state: CloudState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(CloudState)

# Add nodes
workflow.add_node("monitor", monitor_health)
workflow.add_node("diagnose", diagnose_issues)
workflow.add_node("repair", execute_repairs)

# Add edges
workflow.add_edge("monitor", router)
workflow.add_edge("diagnose", router)
workflow.add_edge("repair", router)

# Set entry point
workflow.set_entry_point("monitor")

# Create conditional edges
workflow.add_conditional_edges(
    "monitor",
    router,
    {
        "DIAGNOSE": "diagnose",
        "END": END
    }
)

workflow.add_conditional_edges(
    "diagnose",
    router,
    {
        "REPAIR": "repair",
        "END": END
    }
)

workflow.add_conditional_edges(
    "repair",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [],
        "servers": {
            "web1": {
                "id": "web1",
                "name": "Web Server 1",
                "status": "WARNING",
                "role": "web",
                "current_metrics": {
                    "cpu_usage": 95.0,
                    "memory_usage": 87.0,
                    "disk_usage": 75.0,
                    "network_latency": 150.0,
                    "response_time": 2.5,
                    "error_rate": 2.0,
                    "uptime": 15.5,
                    "timestamp": datetime.now().isoformat()
                },
                "historical_metrics": [],
                "active_issues": [],
                "maintenance_history": []
            }
        },
        "issues": {},
        "resolutions": {},
        "system_health": {},
        "next_action": "MONITOR"
    }
    
    # Run the workflow
    app = workflow.compile()
    for output in app.stream(initial_state):
        print("\nStep Output:")
        print(json.dumps(output, indent=2))
```
{{</ collapse >}}

### 15. ระบบตัดสินใจตามหลักจริยธรรม (Ethical Decision-Making)

รูปแบบนี้มีความสำคัญมากในยุคที่ AI มีบทบาทในการตัดสินใจที่ส่งผลกระทบต่อชีวิตมนุษย์ โดย:

- พิจารณาผลกระทบทางจริยธรรม
- ชั่งน้ำหนักระหว่างประโยชน์และความเสี่ยง
- ตัดสินใจบนพื้นฐานของค่านิยมและบรรทัดฐานของสังคม

รถยนต์ไร้คนขับที่ต้องตัดสินใจในสถานการณ์ฉุกเฉินโดยคำนึงถึงความปลอดภัยของทุกฝ่ายเป็นตัวอย่างที่ดีของการใช้งานรูปแบบนี้

> {{< collapse summary="***ตัวอย่าง Ethical Decision-Making***" >}}
```python
from typing import Dict, List, TypedDict, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenRouter
from enum import Enum
import os

# Define data structures
class EntityType(str, Enum):
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"
    OBSTACLE = "OBSTACLE"
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Position:
    x: float
    y: float
    speed: float
    direction: float

@dataclass
class Entity:
    id: str
    type: EntityType
    position: Position
    size: Tuple[float, float]  # width, height
    velocity: Tuple[float, float]  # vx, vy
    priority: int  # 1 (highest) to 5 (lowest)
    vulnerability: float  # 0-1
    protected_status: bool  # True for children, elderly, etc.

@dataclass
class Scenario:
    timestamp: str
    vehicle_state: Entity
    entities: List[Entity]
    road_conditions: Dict[str, float]  # friction, visibility, etc.
    weather_conditions: Dict[str, str]
    time_to_impact: float
    possible_actions: List[str]

@dataclass
class EthicalPrinciple:
    id: str
    name: str
    description: str
    weight: float
    conditions: List[str]
    priority: int

@dataclass
class Decision:
    action: str
    reasoning: List[str]
    ethical_scores: Dict[str, float]
    risk_assessment: Dict[str, float]
    consequences: List[Dict]
    confidence: float

class VehicleState(TypedDict):
    messages: List[str]
    current_scenario: Dict
    ethical_principles: Dict[str, Dict]
    available_actions: List[str]
    risk_assessments: Dict[str, Dict]
    decision: Optional[Dict]
    next_action: str

# Initialize OpenRouter LLM
llm = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/gpt-4-turbo",
    temperature=0.2,
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Ethical Vehicle AI"
    }
)

# Define ethical principles
ETHICAL_PRINCIPLES = {
    "minimize_harm": EthicalPrinciple(
        id="minimize_harm",
        name="Minimize Harm",
        description="Minimize overall harm to all entities involved",
        weight=1.0,
        conditions=["Consider vulnerability", "Protect human life"],
        priority=1
    ),
    "protect_vulnerable": EthicalPrinciple(
        id="protect_vulnerable",
        name="Protect Vulnerable",
        description="Prioritize protection of vulnerable individuals",
        weight=0.9,
        conditions=["Children", "Elderly", "Disabled"],
        priority=2
    ),
    "fairness": EthicalPrinciple(
        id="fairness",
        name="Fairness",
        description="Ensure fair treatment regardless of characteristics",
        weight=0.8,
        conditions=["No discrimination", "Equal consideration"],
        priority=3
    )
}

class RiskAnalyzer:
    """Analyzes risks for different actions"""
    
    @staticmethod
    def calculate_collision_risk(
        vehicle: Entity,
        entity: Entity,
        time_to_impact: float
    ) -> float:
        # Simple risk calculation based on time to impact and relative velocity
        relative_velocity = (
            (vehicle.velocity[0] - entity.velocity[0])**2 +
            (vehicle.velocity[1] - entity.velocity[1])**2
        )**0.5
        
        distance = (
            (vehicle.position.x - entity.position.x)**2 +
            (vehicle.position.y - entity.position.y)**2
        )**0.5
        
        # Higher risk for closer entities and higher relative velocities
        risk = (relative_velocity * entity.vulnerability) / (distance + 1)
        return min(risk, 1.0)
    
    @staticmethod
    def analyze_action_risks(
        scenario: Scenario,
        action: str
    ) -> Dict[str, float]:
        messages = [
            HumanMessage(content=f"""
            Analyze risks for this action in the current scenario:
            
            Scenario:
            {json.dumps(asdict(scenario), indent=2)}
            
            Proposed Action:
            {action}
            
            Consider:
            1. Collision risks
            2. Entity vulnerabilities
            3. Environmental factors
            4. Time constraints
            
            Provide risk assessment for:
            1. Immediate safety
            2. Secondary effects
            3. Long-term consequences
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Calculate risks (simplified)
        risks = {
            "collision": sum(
                RiskAnalyzer.calculate_collision_risk(
                    scenario.vehicle_state,
                    entity,
                    scenario.time_to_impact
                )
                for entity in scenario.entities
            ) / len(scenario.entities),
            "environmental": sum(
                v for v in scenario.road_conditions.values()
            ) / len(scenario.road_conditions),
            "time_pressure": 1.0 / (scenario.time_to_impact + 1)
        }
        
        return risks

class EthicalEvaluator:
    """Evaluates ethical implications of actions"""
    
    @staticmethod
    def evaluate_action(
        action: str,
        scenario: Scenario,
        principles: Dict[str, EthicalPrinciple],
        risks: Dict[str, float]
    ) -> Dict[str, float]:
        messages = [
            HumanMessage(content=f"""
            Evaluate ethical implications of this action:
            
            Action:
            {action}
            
            Scenario:
            {json.dumps(asdict(scenario), indent=2)}
            
            Ethical Principles:
            {json.dumps({k: asdict(v) for k, v in principles.items()}, indent=2)}
            
            Risk Assessment:
            {json.dumps(risks, indent=2)}
            
            Consider:
            1. Impact on all entities
            2. Adherence to ethical principles
            3. Risk-benefit balance
            4. Social values and norms
            
            Provide scores and explanations for each principle.
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Calculate ethical scores (simplified)
        scores = {}
        for principle in principles.values():
            if principle.id == "minimize_harm":
                scores[principle.id] = 1.0 - max(risks.values())
            elif principle.id == "protect_vulnerable":
                vulnerable_entities = [
                    e for e in scenario.entities
                    if e.protected_status
                ]
                if vulnerable_entities:
                    avg_risk = sum(
                        RiskAnalyzer.calculate_collision_risk(
                            scenario.vehicle_state,
                            entity,
                            scenario.time_to_impact
                        )
                        for entity in vulnerable_entities
                    ) / len(vulnerable_entities)
                    scores[principle.id] = 1.0 - avg_risk
                else:
                    scores[principle.id] = 1.0
            elif principle.id == "fairness":
                # Check if risks are evenly distributed
                risk_variance = max(risks.values()) - min(risks.values())
                scores[principle.id] = 1.0 - risk_variance
        
        return scores

class DecisionMaker:
    """Makes final decisions based on ethical evaluation and risks"""
    
    @staticmethod
    def make_decision(
        scenario: Scenario,
        ethical_scores: Dict[str, Dict[str, float]],
        risk_assessments: Dict[str, Dict[str, float]]
    ) -> Decision:
        messages = [
            HumanMessage(content=f"""
            Make a decision based on ethical evaluation and risks:
            
            Scenario:
            {json.dumps(asdict(scenario), indent=2)}
            
            Ethical Scores:
            {json.dumps(ethical_scores, indent=2)}
            
            Risk Assessments:
            {json.dumps(risk_assessments, indent=2)}
            
            Consider:
            1. Overall ethical alignment
            2. Risk minimization
            3. Time constraints
            4. Practical feasibility
            
            Provide:
            1. Chosen action
            2. Detailed reasoning
            3. Expected consequences
            4. Confidence level
            """)
        ]
        
        response = llm.invoke(messages)
        
        # Select best action (simplified)
        action_scores = {}
        for action in scenario.possible_actions:
            # Combine ethical scores and risk assessments
            ethical_score = sum(
                score * ETHICAL_PRINCIPLES[principle].weight
                for principle, score in ethical_scores[action].items()
            )
            risk_score = 1.0 - sum(risk_assessments[action].values()) / len(risk_assessments[action])
            
            # Weight ethical considerations more heavily
            action_scores[action] = (ethical_score * 0.7) + (risk_score * 0.3)
        
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        return Decision(
            action=best_action,
            reasoning=[
                "Highest combined ethical and safety score",
                f"Ethical score: {ethical_scores[best_action]}",
                f"Risk assessment: {risk_assessments[best_action]}"
            ],
            ethical_scores=ethical_scores[best_action],
            risk_assessment=risk_assessments[best_action],
            consequences=[
                {"type": "immediate", "description": "Avoid collision"},
                {"type": "secondary", "description": "Minimal disruption"}
            ],
            confidence=action_scores[best_action]
        )

def analyze_risks(state: VehicleState) -> VehicleState:
    """
    Analyze risks for each possible action
    """
    # Convert data structures
    scenario = Scenario(**state["current_scenario"])
    
    # Analyze risks for each action
    analyzer = RiskAnalyzer()
    risk_assessments = {}
    
    for action in state["available_actions"]:
        risks = analyzer.analyze_action_risks(scenario, action)
        risk_assessments[action] = risks
    
    state["risk_assessments"] = risk_assessments
    state["next_action"] = "EVALUATE"
    
    return state

def evaluate_ethics(state: VehicleState) -> VehicleState:
    """
    Evaluate ethical implications of each action
    """
    # Convert data structures
    scenario = Scenario(**state["current_scenario"])
    principles = {
        k: EthicalPrinciple(**p)
        for k, p in state["ethical_principles"].items()
    }
    
    # Evaluate each action
    evaluator = EthicalEvaluator()
    ethical_scores = {}
    
    for action in state["available_actions"]:
        scores = evaluator.evaluate_action(
            action,
            scenario,
            principles,
            state["risk_assessments"][action]
        )
        ethical_scores[action] = scores
    
    state["ethical_scores"] = ethical_scores
    state["next_action"] = "DECIDE"
    
    return state

def make_decision(state: VehicleState) -> VehicleState:
    """
    Make final decision based on ethical evaluation and risks
    """
    # Convert data structures
    scenario = Scenario(**state["current_scenario"])
    
    # Make decision
    decision_maker = DecisionMaker()
    decision = decision_maker.make_decision(
        scenario,
        state["ethical_scores"],
        state["risk_assessments"]
    )
    
    state["decision"] = asdict(decision)
    state["next_action"] = "END"
    
    return state

def router(state: VehicleState) -> str:
    """Route to the next step based on the current state"""
    return state["next_action"]

# Create the graph
workflow = StateGraph(VehicleState)

# Add nodes
workflow.add_node("analyze", analyze_risks)
workflow.add_node("evaluate", evaluate_ethics)
workflow.add_node("decide", make_decision)

# Add edges
workflow.add_edge("analyze", router)
workflow.add_edge("evaluate", router)
workflow.add_edge("decide", router)

# Set entry point
workflow.set_entry_point("analyze")

# Create conditional edges
workflow.add_conditional_edges(
    "analyze",
    router,
    {
        "EVALUATE": "evaluate",
        "END": END
    }
)

workflow.add_conditional_edges(
    "evaluate",
    router,
    {
        "DECIDE": "decide",
        "END": END
    }
)

workflow.add_conditional_edges(
    "decide",
    router,
    {
        "END": END
    }
)

# Example usage
if __name__ == "__main__":
    # Initialize state with example emergency scenario
    initial_state = {
        "messages": [],
        "current_scenario": {
            "timestamp": datetime.now().isoformat(),
            "vehicle_state": {
                "id": "ego_vehicle",
                "type": "VEHICLE",
                "position": {
                    "x": 0.0,
                    "y": 0.0,
                    "speed": 50.0,
                    "direction": 0.0
                },
                "size": (2.0, 4.5),
                "velocity": (14.0, 0.0),
                "priority": 3,
                "vulnerability": 0.5,
                "protected_status": False
            },
            "entities": [
                {
                    "id": "pedestrian1",
                    "type": "PEDESTRIAN",
                    "position": {
                        "x": 10.0,
                        "y": 0.0,
                        "speed": 1.0,
                        "direction": 90.0
                    },
                    "size": (0.5, 0.5),
                    "velocity": (0.0, 1.0),
                    "priority": 1,
                    "vulnerability": 0.9,
                    "protected_status": True
                }
            ],
            "road_conditions": {
                "friction": 0.8,
                "visibility": 0.9
            },
            "weather_conditions": {
                "type": "CLEAR",
                "intensity": "NONE"
            },
            "time_to_impact": 0.5,
            "possible_actions": [
                "EMERGENCY_BRAKE",
                "SWERVE_LEFT",
                "SWERVE_RIGHT"
            ]
        },
        "ethical_principles": {
            k: asdict(v) for k, v in ETHICAL_PRINCIPLES.items()
        },
        "available_actions": [
            "EMERGENCY_BRAKE",
            "SWERVE_LEFT",
            "SWERVE_RIGHT"
        ],
        "risk_assessments": {},
        "ethical_scores": {},
        "decision": None,
        "next_action": "ANALYZE"
    }
    
    # Run the workflow
    app = workflow.compile()
    
    print("\nInitial Scenario Analysis:")
    print("=========================")
    print(f"Vehicle Speed: {initial_state['current_scenario']['vehicle_state']['position']['speed']} km/h")
    print(f"Time to Impact: {initial_state['current_scenario']['time_to_impact']} seconds")
    print(f"Available Actions: {', '.join(initial_state['available_actions'])}")
    print("\nStarting Decision Process...")
    
    for output in app.stream(initial_state):
        step = output["next_action"]
        
        if step == "EVALUATE":
            print("\nRisk Assessment Results:")
            print("=======================")
            for action, risks in output["risk_assessments"].items():
                print(f"\nAction: {action}")
                for risk_type, score in risks.items():
                    print(f"- {risk_type}: {score:.2f}")
        
        elif step == "DECIDE":
            print("\nEthical Evaluation Results:")
            print("=========================")
            for action, scores in output["ethical_scores"].items():
                print(f"\nAction: {action}")
                for principle, score in scores.items():
                    print(f"- {principle}: {score:.2f}")
        
        elif step == "END" and output["decision"]:
            print("\nFinal Decision:")
            print("==============")
            decision = output["decision"]
            print(f"Chosen Action: {decision['action']}")
            print("\nReasoning:")
            for reason in decision["reasoning"]:
                print(f"- {reason}")
            print(f"\nConfidence: {decision['confidence']:.2f}")
            print("\nExpected Consequences:")
            for consequence in decision["consequences"]:
                print(f"- {consequence['type']}: {consequence['description']}")
```
The workflow will now execute with detailed outputs at each step

Example Output Explanation:

```text
1. Risk Assessment Phase:
   - Analyzes collision risks for each possible action
   - Considers environmental factors (road conditions, weather)
   - Evaluates time pressure and response windows
   
2. Ethical Evaluation Phase:
   - Applies ethical principles to each action
   - Weighs protection of vulnerable entities
   - Considers fairness and harm minimization
   
3. Decision Making Phase:
   - Combines risk and ethical assessments
   - Selects action with best overall score
   - Provides detailed reasoning and expected outcomes

The system prioritizes:
- Protection of human life
- Minimization of harm
- Fairness in risk distribution
- Consideration of vulnerable individuals
- Practical feasibility of actions

Key ethical principles like minimizing harm and protecting vulnerable individuals 
are given higher weights in the decision process, while still maintaining 
a balance with practical safety considerations.
```
{{</ collapse >}}

## การนำ Agentic Design Patterns ไปใช้งาน

การเลือกใช้รูปแบบการออกแบบที่เหมาะสมเป็นสิ่งสำคัญมาก เพราะแต่ละรูปแบบมีจุดแข็งและข้อจำกัดที่แตกต่างกัน ในการพัฒนาระบบ AI ควรพิจารณาปัจจัยต่างๆ ดังนี้:

1. **ลักษณะของงาน**: งานที่ต้องการการตอบสนองรวดเร็วอาจเหมาะกับ Reflexive Agent ในขณะที่งานที่ซับซ้อนอาจต้องใช้ Meta-Agent หรือ Collaborative Multi-Agent Systems

2. **ทรัพยากรที่มี**: บางรูปแบบต้องการทรัพยากรการประมวลผลมาก เช่น Self-Improvement หรือ Adaptive Workflow Orchestration ควรพิจารณาความพร้อมของระบบก่อนเลือกใช้

3. **ความต้องการด้านความแม่นยำ**: งานที่ต้องการความแม่นยำสูงอาจต้องใช้รูปแบบที่มีการตรวจสอบและยืนยันผลลัพธ์ เช่น ReACT หรือ Planner-Executor

4. **ความต้องการด้านการปรับตัว**: หากระบบต้องทำงานในสภาพแวดล้อมที่เปลี่ยนแปลงบ่อย ควรเลือกรูปแบบที่มีความยืดหยุ่นสูง เช่น Self-Improvement หรือ Interactive Learning

## บทสรุป

Agentic Design Patterns เป็นแนวคิดที่น่าสนใจและมีประโยชน์มากในการพัฒนาระบบ AI ให้ทำงานได้อย่างชาญฉลาด การเข้าใจจุดแข็งและข้อจำกัดของแต่ละรูปแบบจะช่วยให้เราสามารถเลือกใช้และผสมผสานรูปแบบต่างๆ ได้อย่างเหมาะสม เพื่อสร้างระบบ AI ที่มีประสิทธิภาพและตอบโจทย์ความต้องการได้อย่างแท้จริง

## แหล่งข้อมูลเพิ่มเติม

- [Agentic Design Patterns](https://github.com/panaversity/learn-agentic-ai/tree/main/05_ai_agents_intro/13_agentic_design_patterns)

---

*Cover image by [AI Agentic Design Patterns with AutoGen](https://github.com/ksm26/AI-Agentic-Design-Patterns-with-AutoGen)*

*ปล. บทความนี้เขียนด้วย AI  (^ . ^)*