---
author: "Apiwat Ruangkanjanapaisarn"
title: "ลองเล่น Local LLM ด้วย Ollama + Python"
date: 2025-02-02
draft: false
description: "มาเรียนรู้การใช้งาน Local LLM ผ่าน Ollama ร่วมกับ Python เพื่อสร้าง AI Application แบบ Privacy-First"
tags: ["AI", "LLM", "Python", "Ollama", "Local Development"]
categories: ["Artificial Intelligence", "Development"]
cover:
  image: https://ollama.com/public/blog/embedding-models.png
---

# ลองเล่น Local LLM ด้วย Ollama + Python

*บทความนี้จะพาทุกคนมาลองใช้งาน LLM บนเครื่องคอมพิวเตอร์ส่วนตัวผ่าน Ollama และ Python เหมาะสำหรับผู้ที่อยากทดลองเล่น AI แต่กังวลเรื่องความเป็นส่วนตัวของข้อมูล หรือต้องการระบบที่ทำงานได้แม้ไม่มีอินเทอร์เน็ต*

## ที่มาของ Large Language Model (LLM)

ในช่วงไม่กี่ปีที่ผ่านมา เราได้เห็นการเติบโตอย่างก้าวกระโดดของ AI โดยเฉพาะในด้านการประมวลผลภาษาธรรมชาติ จุดเปลี่ยนสำคัญเกิดขึ้นเมื่อนักวิจัยพบว่า การสร้างโมเดลขนาดใหญ่และฝึกฝนด้วยข้อมูลมหาศาล ทำให้ AI สามารถเข้าใจและตอบโต้กับมนุษย์ได้อย่างน่าทึ่ง

ปัจจุบันมีบริการ LLM มากมายให้เลือกใช้ เช่น ChatGPT, Claude, Gemini แต่หลายคนอาจกังวลเรื่องความเป็นส่วนตัวของข้อมูล หรือต้องการระบบที่ทำงานได้แม้ไม่มีอินเทอร์เน็ต นั่นคือที่มาของ Local LLM

## รู้จักกับ Ollama

Ollama เป็นเครื่องมือที่ช่วยให้เราสามารถรัน LLM บนเครื่องคอมพิวเตอร์ส่วนตัวได้อย่างง่ายดาย รองรับโมเดลหลากหลาย เช่น Llama 3, Mistral, CodeLlama โดยมีจุดเด่นคือ:

- ติดตั้งง่าย รองรับทั้ง Windows, macOS และ Linux
- มี API ที่ใช้งานสะดวก
- ประสิทธิภาพดี ใช้ทรัพยากรเครื่องน้อย
- รองรับการปรับแต่งโมเดลได้ตามต้องการ

## การติดตั้ง

### 1. ติดตั้ง Ollama

สำหรับ macOS:
```bash
brew install ollama
```

สำหรับ Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

สำหรับ Windows สามารถดาวน์โหลดได้จาก [เว็บไซต์ Ollama](https://ollama.com)

### 2. ติดตั้ง Python Package

```bash
pip install ollama
```

## เริ่มต้นใช้งาน

### 1. ดาวน์โหลดโมเดล

เริ่มจากเปิด Terminal แล้วรันคำสั่ง:

```bash
ollama pull llama3.1
```

### 2. ทดสอบด้วย Python

สร้างไฟล์ `test_ollama.py`:

```python
import ollama

def simple_chat():
    response = ollama.chat(model='llama3.1', 
                          messages=[
                              {'role': 'user', 
                               'content': 'สวัสดี คุณทำอะไรได้บ้าง?'}
                          ])
    print(response['message']['content'])

# ทดสอบเรียกใช้งาน
if __name__ == '__main__':
    simple_chat()
```

ลองรันทดสอบ:

```bash
python test_ollama.py
```

Output:
```
สวัสดีค่ะ ฉันสามารถตอบคำถามของคุณได้ เช่น การเรียนรู้ภาษา คำนวณเลขคณิต ช่วยหาข้อมูลเกี่ยวกับประเทศหรือเมือง ขอข้อมูลเกี่ยวกับต่างๆ อีกมากมายค่ะ
```

## การใช้งานขั้นสูงขึ้น

### การสร้าง Chat Assistant

สร้างไฟล์ `assistant.py`:

```python
import ollama

from typing import List, Dict

class ChatAssistant:
    def __init__(self, model_name: str = 'llama3.1'):
        self.model = model_name
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat(self, message: str) -> str:

        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        response = ollama.chat(
            model=self.model,
            messages=self.conversation_history
        )
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response['message']['content']
        })
        
        return response['message']['content']
    
    def clear_history(self):
        self.conversation_history = []
```

### ตัวอย่างการใช้งาน Chat Assistant

สร้างไฟล์ `chat.py`:

```python
from assistant import ChatAssistant

assistant = ChatAssistant()

questions = [
    "Python คืออะไร?",
    "ยกตัวอย่างการใช้งาน list comprehension",
    "แล้ว dictionary comprehension ล่ะ?"
]

for question in questions:
    print(f"\nคำถาม: {question}")
    print(f"คำตอบ: {assistant.chat(question)}")
```

ลองรันทดสอบ:

```bash
python chat.py
```

Output:
```
คำถาม: Python คืออะไร?
คำตอบ: ภาษาเชิงสคริปต์ (Scripting language) ที่ใช้ในการเขียนโปรแกรมคอมพิวเตอร์ โดยมีลักษณะเฉพาะคือความสามารถในการนำโค้ดไปใช้งานได้ทันทีโดยไม่ต้องบันทึกลงไปในไฟล์ใดๆ

คำถาม: ยกตัวอย่างการใช้งาน list comprehension
คำตอบ: **List Comprehension ในภาษา Python**

List comprehension เป็นฟังก์ชันพิเศษในภาษา Python ที่สามารถสร้างรายการ (list) ได้อย่างรวดเร็วและง่ายดาย โดยไม่ต้องใช้ loop หรือการเขียนโค้ดซ้ำๆ

ตัวอย่างการใช้งาน list comprehension:

**1. สร้างรายการที่มีขนาดเฉพาะ**

`python
numbers = [i for i in range(10)]
print(numbers)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
`

**2. ฟิลเตอร์รายการ**

`python
numbers = [i for i in range(10) if i % 2 == 0]
print(numbers)  # [0, 2, 4, 6, 8]
`

**3. ทำการปฏิบัติการบนรายการ**

`python
numbers = [i ** 2 for i in range(5)]
print(numbers)  # [0, 1, 4, 9, 16]
`

**4. รวมสองรายการเข้าด้วยกัน**

`python
names = ['John', 'Alice', 'Bob']
ages = [25, 30, 35]

people = [{name: age} for name, age in zip(names, ages)]
print(people)  
# [{'John': 25}, {'Alice': 30}, {'Bob': 35}]
`

นี่คือตัวอย่างการใช้งาน list comprehension ในภาษา Python มีหลายกรณีที่สามารถใช้ได้ และมันช่วยให้คุณเขียนโค้ดที่กระชับและง่ายดายมากขึ้น!

คำถาม: แล้ว dictionary comprehension ล่ะ?
คำตอบ: **Dictionary Comprehension ในภาษา Python**

 Dictionary comprehension เป็นฟังก์ชันพิเศษในภาษา Python ที่สามารถสร้าง辞านวารี (dictionary) ได้อย่างรวดเร็วและง่ายดาย โดยไม่ต้องใช้ loop หรือการเขียนโค้ดซ้ำๆ

ตัวอย่างการใช้งาน dictionary comprehension:

**1. สร้าง辞านวารีที่มีขนาดเฉพาะ**

`python
numbers = {i: i * 2 for i in range(5)}
print(numbers)  
# {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
`

**2. ฟิลเตอร์รายการ**

`python
numbers = {i: i * 2 for i in range(10) if i % 2 == 0}
print(numbers)  
# {0: 0, 2: 4, 4: 8, 6: 12, 8: 16}
`

**3. ทำการปฏิบัติการบนรายการ**

`python
numbers = {i: i ** 2 for i in range(5)}
print(numbers)  
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
`

**4. รวมสองรายการเข้าด้วยกัน**

`python
names = ['John', 'Alice', 'Bob']
ages = [25, 30, 35]

people = {name: age for name, age in zip(names, ages)}
print(people)  
# {'John': 25, 'Alice': 30, 'Bob': 35}
`

นี่คือตัวอย่างการใช้งาน dictionary comprehension ในภาษา Python มีหลายกรณีที่สามารถใช้ได้ และมันช่วยให้คุณเขียนโค้ดที่กระชับและง่ายดายมากขึ้น!

ความแตกต่างระหว่าง list comprehension และ dictionary comprehension คือ:

* List comprehension สร้างรายการ (list) ขณะที่ dictionary comprehension สร้าง辞านวารี (dictionary)
* ใน list comprehension เราสามารถใช้คำสั่ง `for` ได้ทั้งสองฝ่าย (left-hand side และ right-hand side) ในขณะที่ใน dictionary comprehension เราสามารถใช้คำสั่ง `for` ได้เพียงฝ่ายหนึ่งเท่านั้น
```
## การปรับแต่งพารามิเตอร์

เราสามารถปรับแต่งการทำงานของ LLM ได้ผ่านพารามิเตอร์ต่างๆ:

สร้างไฟล์ `advanced_chat.py`:

```python
import ollama

def advanced_chat(prompt: str):
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'temperature': 0.7,  # ควบคุมความสร้างสรรค์ (0.0 - 1.0)
            'top_p': 0.9,       # ควบคุมความหลากหลายของคำตอบ
            'top_k': 40,        # จำนวนโทเค็นที่พิจารณา
            'num_predict': 4069  # ความยาวสูงสุดของคำตอบ
        }
    )
    return response['message']['content']

# ทดสอบเรียกใช้งาน
if __name__ == '__main__':
    prompt = "เล่าเรื่องตลกให้ฟังหน่อยสิ"
    print(advanced_chat(prompt))
```

ลองรันทดสอบ:

```bash
python advanced_chat.py
```

Output:
```
มีชายคนหนึ่งซื้อหมูจากตลาดกลับบ้านเพื่อให้ทานเย็น แต่เมื่อลูกสาวของเขาเห็นหมู เธอก็บอกพ่อว่า "พ่อ ฉันอยากจะเลี้ยงหมูตัวนั้นก่อน"

ชายคนนั้นพยายามที่จะทำให้ลูกสาวตกใจและบอกเธอว่า "หมูนี้เป็นหมูที่มีชื่อเสียงมาก มันสามารถปรุงแต่งอาหารได้ทุกชนิด แต่สิ่งที่สำคัญที่สุดคือมันไม่ต้องการเงิน"

หญิงสาวตอบว่า "นั่นก็ทำให้ฉันประหลาดใจจริงๆ ที่เราสามารถจ่ายค่าตอบแทนทางเงินให้มันได้!"
```

## การใช้งานกับ Stream

Ollama รองรับการ stream ข้อความตอบกลับแบบ real-time:

```python
import ollama

def stream_chat(prompt: str):
    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    
    # พิมพ์ข้อความทีละส่วนตามที่ได้รับ
    for chunk in stream:
        if chunk['message']['content']:
            print(chunk['message']['content'], end='', flush=True)
```

## การจัดการกับข้อผิดพลาด

```python
import ollama

def safe_chat(prompt: str) -> str:
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"
```

## ข้อควรระวังและข้อจำกัด

1. ทรัพยากรเครื่อง
   - ต้องการ RAM อย่างน้อย 8GB
   - ควรมี GPU สำหรับประสิทธิภาพที่ดี
   - พื้นที่ดิสก์สำหรับเก็บโมเดล (ประมาณ 4-8GB ต่อโมเดล)

2. ความแม่นยำ
   - Local LLM อาจมีความแม่นยำน้อยกว่าโมเดลออนไลน์
   - ควรตรวจสอบผลลัพธ์เสมอ โดยเฉพาะในงานสำคัญ

3. การอัพเดท
   - ติดตามการอัพเดทของ Ollama และโมเดลอยู่เสมอ
   - อาจต้อง pull โมเดลใหม่เมื่อมีเวอร์ชันอัพเดท

## สรุป

การใช้ Local LLM ผ่าน Ollama เป็นทางเลือกที่น่าสนใจสำหรับผู้ที่ต้องการความเป็นส่วนตัวหรือต้องการระบบที่ทำงานได้แบบ offline ถึงแม้จะมีข้อจำกัดบางประการ แต่ก็สามารถนำไปประยุกต์ใช้ได้หลากหลาย ตั้งแต่การสร้าง chatbot ไปจนถึงการประมวลผลเอกสาร

## แหล่งข้อมูลเพิ่มเติม

- [GitHub Repo](https://github.com/toffysoft/ollama-python-example)
- [Ollama Official Documentation](https://ollama.com/docs)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Python Package Documentation](https://github.com/ollama/ollama-python)

---

*บทความนี้อัพเดทล่าสุด: กุมภาพันธ์ 2025*

*Note: ตัวอย่างโค้ดทั้งหมดทดสอบบน Python 3.10+*

*Cover image by [Ollama](https://ollama.com)*

*ปล. บทความนี้เขียนด้วย AI  (^ . ^)*
