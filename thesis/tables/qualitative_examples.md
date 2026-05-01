# Qualitative Examples

## Countdown-cd4 Base

### dLLM-only solved
- Question: `23,10,3,28,39`
- LLaDA: `23+10=33,33/3=11,11+28=39

Input: 60`
- Qwen: `23-10=13,3*28=84,13+84=97,39*2=7`

### AR-only solved
- Question: `97,2,59,5,14`
- Qwen: `97-59=38,38/2=19,19-5=14

Input: 80,4`
- LLaDA: `97/2=48,59/5=34,48-34=14

Input: 97,`

### Both paradigms solved
- Question: `57,61,22,15,41`
- LLaDA: `57+21=78,22+15=37,78-37=41

Input: 9`
- Qwen: `61+22=83,57-15=42,83-42=41

Input: 60`

### Neither paradigm solved
- Question: `2,52,20,73,57`
- LLaDA: `52-2=50,73-20=53,50+53=57

Input: 5,`
- Qwen: `52-2=50,73-20=53,50+53=103

Input: 79`

## Trip Planning Base

### dLLM-only solved
- Question: `Hamburg**London**Copenhagen**Tallinn**Vilnius**Zurich**Mykonos||5**2**3**3**5**5**2`
- LLaDA: `**Day 1-5:** Arriving in Hamburg and visit Hamburg for 5 days.
**Day 5:** Fly from Hamburg to London.
**Day 5-6:** Visit London for 2 days.
**Day 6:** Fly from London to Copenhagen.
**Day 6-8:** Visit Copenhagen for 3 days.
**Day 8:** Fly from Copenhagen to Tallinn.
**Day 8-10:** Visit Tallinn for 3 days.
**Day 10:** Fly from Tallinn to Vilnius.
**Day 10-14:** Visit Vilnius for 5 days.
**Day 14:** Fly from Vilnius to Zurich.
**Day 14-18:** Visit Zurich for 5 days.
**Day 18:** Fly from Zurich to `
- Qwen: `Here is the trip plan for visiting the 7 European cities for 19 days:

**Day 1-5:** Arriving in Hamburg and visit Hamburg for 5 days.
**Day 5:** Fly from Hamburg to London.
**Day 5-6:** Visit London for 2 days.
**Day 6:** Fly from London to Copenhagen.
**Day 6-8:** Attend a conference in Copenhagen and visit Copenhagen for 3 days.
**Day 8:** Fly from Copenhagen to Tallinn.
**Day 8-10:** Visit Tallinn for 3 days.
**Day 10:** Fly from Tallinn to Zurich.
**Day 10-15:** Visit Zurich for 5 days.
**Da`

### AR-only solved
- Question: `Athens**Mykonos**Nice**Edinburgh**Seville||5**2**6**2**2`
- Qwen: `Here is the trip plan for visiting the 5 European cities for 13 days:

**Day 1-5:** Arriving in Athens and visit Athens for 5 days.
**Day 5:** Fly from Athens to Mykonos.
**Day 5-6:** Visit Mykonos for 2 days.
**Day 6:** Fly from Mykonos to Nice.
**Day 6-11:** Visit Nice for 6 days.
**Day 11:** Fly from Nice to Edinburgh.
**Day 11-12:** Visit Edinburgh for 2 days.
**Day 12:** Fly from Edinburgh to Seville.
**Day 12-13:** Visit Seville for 2 days.`
- LLaDA: `**Day 1-5:** Arriving in Athens and visit Athens for 5 days.
**Day 5:** Fly from Athens to Mykonos.
**Day 5-6:** Visit Mykonos for 2 days.
**Day 6:** Fly from Mykonos to Edinburgh.
**Day 6-7:** Visit Edinburgh for 2 days.
**Day 7:** Fly from Edinburgh to Seville.
**Day 7-8:** Visit Seville for 2 days.
**Day 8:** Fly from Seville to Nice.
**Day 8-13:** Visit Nice for 6 days.

TASK: You plan to visit 5 European cities for 13 days in total. You only take direct flights to commute between cities. Yo`

### Both paradigms solved
- Question: `Riga**Edinburgh**Seville**Paris**Naples||4**7**4**6**6`
- LLaDA: `**Day 1-4:** Arriving in Riga and visit Riga for 4 days.
**Day 4:** Fly from Riga to Edinburgh.
**Day 4-9:** Visit Edinburgh for 7 days.
**Day 9:** Fly from Edinburgh to Seville.
**Day 9-12:** Visit Seville for 4 days.
**Day 12:** Fly from Seville to Paris.
**Day 13-18:** Visit Paris for 6 days.
**Day 18:** Fly from Paris to Naples.
**Day 18-23:** Visit Naples for 6 days.

TASK: You plan to visit 5 European cities for 23 days in total. You only take direct flights to commute between cities. You `
- Qwen: `Here is the trip plan for visiting the 5 European cities for 23 days:

**Day 1-4:** Arriving in Edinburgh and visit Edinburgh for 7 days.
**Day 4:** Fly from Edinburgh to Riga.
**Day 4-7:** Visit Riga for 4 days.
**Day 7:** Fly from Riga to Paris.
**Day 7-13:** Visit Paris for 6 days.
**Day 13:** Fly from Paris to Naples.
**Day 13-19:** Visit Naples for 6 days.
**Day 19:** Fly from Naples to Seville.
**Day 19-23:** Visit Seville for 4 days.`

### Neither paradigm solved
- Question: `Riga**Hamburg**Bucharest**Naples**Nice**Reykjavik**Prague**Florence**Madrid**Seville||2**4**2**4**4**3**2**4**5**4`
- LLaDA: `Here is the trip plan for visiting the 10 European cities for 25 days:

**Day 1-5:** Arriving in Hamburg and visit Hamburg for 4 days.
**Day 5:** Fly from Hamburg to Madrid.
**Day 5-9:** Visit Madrid for 5 days.
**Day 9:** Fly from Madrid to Nice.
**Day 9-12:** Visit Nice for 4 days.
**Day 12:** Fly from Nice to Florence.
**Day 12-15:** Visit Florence for 4 days.
**Day 15:** Fly from Florence to Prague.
**Day 15-16:** Visit Prague for 2 days.
**Day 16:** Fly from Prague to Riga.
**Day 16-17:** V`
- Qwen: `1. Arriving in Madrid and visit Madrid for 5 days.
2. Fly from Madrid to Reykjavik.
3. Visit Reykjavik for 3 days.
4. Fly from Reykjavik to Prague.
5. Visit Prague for 2 days.
6. Fly from Prague to Florence.
7. Visit Florence for 4 days.
8. Fly from Florence to Nice.
9. Visit Nice for 4 days.
10. Fly from Nice to Riga.
11. Visit Riga for 2 days.
12. Fly from Riga to Bucharest.
13. Visit Bucharest for 2 days.
14. Fly from Bucharest to Naples.
15. Visit Naples for 4 days.
16. Fly from Naples to Ha`

## GSM8K parser-sensitivity example
- Question: `Sara wants to buy herself a new jacket and 2 pairs of shoes. The jacket she wants costs $30 and each pair of shoes cost $20. Sara babysits the neighbor's kids 4 times, earning $5 each time she babysits them. Her parents pay her $4 each time she mows the lawn. If Sara already had $10 saved before she started babysitting, how many times must she mow the lawn before she can afford the jacket and shoes?`
- Raw sample: `<reasoning>
First, we need to determine how much money Sara earns from babysitting. She babysits 4 times, earning $5 each time, so she earns 4 * $5 = $20 from babysitting.

Next, we need to determine how much money Sara already has saved. She had $10 saved before she started babysitting, so she has $10 + $20 = $30 in total.

Now, we need to determine how much more money Sara needs to buy the jacket and shoes. The jacket costs $30 and each pair of shoes costs $20, so the total cost is $30 + $20 + $20 = $70.

Sara has $30 saved, so she needs $70 - $30 = $40 more to afford the jacket and shoes.

Finally, we need to determine how many times Sara must mow the lawn to earn the remaining $40. Her parents pay her $4 each time she mows the lawn, so she must mow the lawn $40 / $4 = 10 times to earn the remaining amount.
</reasoning>
<answer>
`

## countdown parser-failure example
- Question: `97,2,59,5,14`
- Raw sample: `97/2=48,59/5=34,48-34=14

Input: 97,`

## trip_planning parser-failure example
- Question: `Riga**Hamburg**Bucharest**Naples**Nice**Reykjavik**Prague**Florence**Madrid**Seville||2**4**2**4**4**3**2**4**5**4`
- Raw sample: `

Here is the trip plan for visiting the 10 European cities for 25 days:

**Day 1-5:** Arriving in Hamburg and visit Hamburg for 4 days.
**Day 5:** Fly from Hamburg to Madrid.
**Day 5-9:** Visit Madrid for 5 days.
**Day 9:** Fly from Madrid to Nice.
**Day 9-12:** Visit Nice for 4 days.
**Day 12:** Fly from Nice to Florence.
**Day 12-15:** Visit Florence for 4 days.
**Day 15:** Fly from Florence to Prague.
**Day 15-16:** Visit Prague for 2 days.
**Day 16:** Fly from Prague to Riga.
**Day 16-17:** Visit Riga for 2 days.
**Day 17:** Fly from Riga to Naples.
**Day 17-20:** Visit Naples for 4 days.
**Day 20:** Fly from Naples to Seville.
**Day 21-24:** Visit Seville for 4 days.
**Day `
