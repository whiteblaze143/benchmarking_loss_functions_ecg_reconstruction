
# Post-EMBC Meeting Script

## 1. Open naturally — 1–2 minutes

**You:**

“Thanks again for making the time. I know things have been pretty hectic with the grants and everything.

I mainly wanted to use this as a chance to debrief EMBC while it's still fresh, get your thoughts on where the ECG work should go next, and also talk a little more broadly about how you see some of these projects developing over the next year or two.”

Then stop.

Let him respond. Do not launch directly into a ten-minute presentation.

If he asks how EMBC was:

“It was really useful. Beyond presenting the work, probably the biggest value was seeing how other groups are thinking about the clinical utility problem. I came back with a slightly different view of what the next question should be.”

---

# 2. Give the EMBC research update — 3–4 minutes

“I think the architecture and loss-function work gave us a good foundation. We've now looked pretty systematically across ECG-AIM, the multiscale VAE and U-Net, and across the different loss combinations.

What stood out to me is that we're probably getting close to diminishing returns if the next project is just another model that improves correlation by a few points.

After the conference I spent some time looking at the reconstruction literature more critically, especially the Presacan paper. Their argument is basically that you can have a reconstruction that looks excellent and has good correlation, but it can still regress toward population-average morphology rather than preserve what is actually patient-specific.

We haven't directly tested that in our models yet.”

### If Chris looks interested:

“So one thing I think we should do immediately is a regression-to-the-mean audit on what we already have—Bland–Altman, variance preservation, calibration slope, error versus true amplitude, and performance in the morphology tails. That's relatively low-cost and would tell us whether the good reconstruction metrics are actually telling the full story.”

### Ask him:

“From your perspective clinically, which measurements would you care most about preserving exactly? QT/QTc, QRS duration, R/S amplitudes, ST/J-point morphology, something else?”

Then listen.

This is an important point where you want **Chris to define the clinical endpoint**.

---

# 3. Bring up the Stanford work — 4–5 minutes

“One of the more interesting conversations I had around the conference was with the Stanford group.

What I thought they did particularly well wasn't just the reconstruction itself. They anchored the whole problem to something clinically actionable—QT monitoring after class III antiarrhythmic initiation.

They weren't trying to prove that every synthetic lead was interchangeable with a clinical ECG. They were asking whether the information needed for QT/QTc could be recovered accurately enough from an ICM signal to identify patients at risk.”

Then make the key transition:

“That made me think our next question probably shouldn't be, ‘Can we reconstruct twelve leads better?’ It should be, ‘What information do we want from the reconstructed ECG that we cannot already get from the limited lead, and would that information actually change care?’”

Pause.

Then:

“I've kept talking with Rayyan, the grad student there. We've been bouncing around some of the validation and physiology questions. There might eventually be a useful collaboration there, but before pushing that further I wanted to get your view on what would actually complement what we're doing here.”

### Ask Chris:

“Do you know the Stanford investigators involved in that work, or have you worked with anyone in that group before?”

Then:

“More broadly, are there people you've collaborated with—at U of T, UHN, elsewhere in EP, or even outside Toronto—who you think would be particularly useful for this kind of reconstruction/clinical-validation question?”

This is a better way to ask about collaborators than:

> “Who can I work with?”

You're asking him to help design the scientific team.

---

# 4. Transition into Fitbit — 7–10 minutes

“I've also been thinking about whether there is a natural connection to the Fitbit study.

My first thought was AF recurrence, but the more I think about it, the more cautious I am about forcing reconstruction into that problem because AF itself is already identifiable from a single lead.”

Then ask:

“If we could recover additional ECG information from the Fitbit reliably, what information would actually be useful to you in these patients beyond simply detecting AF?”

Let Chris answer before suggesting possibilities.

If necessary, prompt with:

“Would differentiating recurrent AF from flutter or atrial tachycardia be useful?”

“Would QT/QTc surveillance ever matter in this cohort based on their antiarrhythmic use?”

“Are there conduction or repolarization changes you would care about longitudinally?”

“Or is the answer realistically that 12-lead reconstruction doesn't add enough to the AF recurrence question?”

If he says **reconstruction probably does not add much**:

“I think that's useful to know. I'd rather keep the studies separate than create an artificial connection just because we have both datasets.”

If he identifies a useful ECG quantity:

“That gives us a much cleaner target. Then the model can be designed and evaluated around that endpoint rather than around generic waveform similarity.”

---

# 5. Ask about paired Fitbit + 12-lead acquisition

“One thing that could change the value of the Fitbit study for this question is paired data.

Do patients already get clinical 12-leads around enrollment, the procedure, or follow-up visits?”

If yes:

“Would it be operationally feasible to have them take a Fitbit ECG at essentially the same time?”

Then:

“Even a relatively small paired subset could be much more informative than another large retrospective benchmark because we'd have the actual wearable vector and contemporaneous clinical ground truth from the same patient.”

### If Chris says it would require amendment / extra coordination:

“That's what I assumed. I don't think it's something we should add unless it answers a sufficiently important clinical question. But I wanted to know whether the opportunity exists before we decide on the next modelling direction.”

That response shows restraint.

---

# 6. Talk about the technically ambitious direction only after the clinical discussion

“I've also been thinking about the technical side, but I don't want to choose the model before we've chosen the question.

One direction I find interesting is moving away from deterministic reconstruction. If limited leads don't uniquely determine the missing precordial morphology, forcing the network to output one waveform is part of what creates the population-average problem.

We could instead model a patient-specific latent cardiac state and a distribution of possible reconstructions, where the system can actually say when a missing lead is poorly identifiable.”

If he's interested:

“The more ambitious version would incorporate physiological constraints or a world-model-type formulation. But I don't think the contribution should be ‘we used a world model.’ The useful part would be estimating the latent cardiac state, representing uncertainty and potentially saying: given the leads I currently have, which additional measurement would reduce uncertainty most?”

Then stop.

You do **not** need to explain latent diffusion, ODE energy guidance, active sensing, etc. unless he asks.

---

# 7. Transition into your longer-term research plans

After the research direction has been discussed, change gears explicitly:

“There was one other thing I wanted to ask you about while we're talking about the longer-term direction.”

Pause.

“I've been thinking pretty seriously about what I want my next stage to look like after my current training, and I'm increasingly interested in doing something where I can stay quite deeply involved in clinical AI and electrophysiology rather than treating these as isolated projects.”

Then:

“I wanted to ask how you see your research program developing over the next couple of years.”

This is deliberately open-ended.

Listen for:

* new grants;
* Fitbit expansion;
* wearable research;
* ECG AI;
* TAVR/device projects;
* industry partnerships;
* new trainees;
* faculty collaborators;
* data infrastructure.

Then ask:

“Are there particular projects that you think are likely to become major parts of the group rather than shorter individual studies?”

---

# 8. Ask the PhD question intelligently

Do **not** start with:

> “Can you supervise my PhD?”

Start by understanding the landscape.

“I've also been considering PhD programs fairly seriously. If I stayed in this area, I'd want it to be somewhere with strong clinical mentorship but also genuinely strong technical supervision.”

Then:

“How does PhD supervision normally work from your side? Would you typically co-supervise with someone in engineering or computer science?”

This naturally opens the Alex question without putting anyone on the spot.

Then:

“If you were advising me, are there people you've worked with who you think would make particularly strong PhD supervisors or co-supervisors for someone trying to sit right at the clinical EP/ML intersection?”

Follow with:

“Would you see something with you and Alex making sense, or do you think there's another configuration that would be stronger?”

That formulation is excellent because you are **asking for his judgment**, not lobbying.

---

# 9. Ask what he thinks makes you competitive

This is valuable information and also invites mentorship.

“If I wanted to put myself in a strong position for PhD applications over the next year, what do you think would matter most?”

Then give him options only if needed:

“Is it primarily publications? A stronger methodological paper? More prospective clinical work? Getting involved with a larger collaboration?”

This may reveal what he thinks of your trajectory.

If he says you need stronger publications:

“That's helpful. Then I probably want to be deliberate about turning this next ECG project into something deeper than another conference abstract.”

If he mentions MICCAI / journal / larger venue:

“That's aligned with what I've been thinking. I'd rather spend the time answering a harder clinical-validity question properly than maximize the number of incremental papers.”

---

# 10. Ask about funding directly, but professionally

Once PhD supervision is being discussed:

“How does funding usually work for your trainees?”

Then:

“Do you currently have grants—or applications in progress—that could support a graduate student working in this area?”

Because Chris already mentioned being busy with grants, this is a natural question.

You can also ask:

“Would funding typically come through your grants, the technical co-supervisor, departmental funding, scholarships, or some combination?”

Then:

“Are there specific scholarships or programs you think I should be positioning myself for now?”

This is strategically much better than asking:

> “Can you fund me?”

You're learning the mechanism.

---

# 11. Ask about full-time work beginning summer 2027

Do this after the PhD discussion, because the two possibilities are related.

“I also wanted to ask about something a little more near-term because I'm planning fairly far ahead.

Starting next summer, I should have substantially more flexibility, and I'd be very interested in spending a period working on these projects essentially full-time if there were a meaningful role and funding available.”

Then:

“Do you think there is any realistic possibility of a full-time research position with your group starting around summer 2027?”

Stop talking.

Let him answer.

### If he says funding is uncertain:

“That's completely fair. I'm not asking for a commitment now—I mainly wanted to understand whether it's something worth planning around, and what would need to happen between now and then for it to become feasible.”

Then:

“Is there a particular grant decision or funding cycle that would determine that?”

Excellent question.

### If he says yes / potentially:

“That would be very interesting to me. What kind of role would you see being most useful—driving the ECG/AI work, helping run the Fitbit study, broader analytics across the EP projects, or some combination?”

### If he says probably not:

“No problem. That's useful for me to know early. I'd still like to keep building the research with you; I just want to plan the next year realistically.”

Do not show disappointment.

---

# 12. If he asks, “What are your longer-term plans?”

This is where you should be truthful but selective.

Say:

“I'm keeping both doctoral research and medicine open long-term. I like the clinical side enough that I don't want to lose that, but I also really enjoy doing the technical research at a level where I'm building the methods rather than only applying them.

Because of timing, I'm not applying to medicine in this immediate cycle, so I have a useful window where I can invest seriously in research. If I find the right PhD environment and problem, I'd be very open to doing the PhD first.”

That communicates everything Chris needs to know.

You **do not need to say**:

* “I haven't written my MCAT yet.”
* “If I don't get a good PhD I'll study for the MCAT.”
* “Working with you is my backup.”
* “I'm trying to hedge between careers.”

Those facts do not help him answer the question you are asking.

If he pushes further:

“I don't feel pressure to decide between medicine and research immediately. What I care about over the next year is putting myself somewhere where I'm learning a lot, producing strong work, and getting enough exposure to know which path I want to commit to.”

---

# 13. Ask Chris about his own plans

This is important and easy to overlook.

“I've talked a lot about what I'm considering, but I'm also curious about what you're planning. Where do you want to take the research side of your work over the next few years?”

Then listen.

Possible follow-ups:

“Are you hoping to grow the AI/wearables side substantially?”

“Are you planning to take on more graduate students?”

“Are there grants you're pursuing specifically around digital health or AI?”

“Do you see the Fitbit study becoming a larger platform for other questions?”

“Are there other datasets or prospective studies you're hoping to build?”

This gives you intelligence about whether there is actually a future research ecosystem worth joining.

---

# 14. Ask about collaborators and mentorship

If the PhD discussion is positive:

“One thing I want to be thoughtful about is mentorship. Who have you worked with where you thought the collaboration between the clinical and technical sides worked particularly well?”

Then:

“If you were putting together the ideal supervisory committee for this kind of PhD, who would you want represented?”

Potential categories:

* electrophysiology;
* machine learning;
* biomedical engineering;
* signal processing;
* digital health / HCI;
* statistics / clinical trials.

Then:

“Would you be comfortable introducing me to some of those people once the research direction is clearer?”

That is a concrete but reasonable ask.

---

# 15. If Chris proposes an idea during the meeting

Do not immediately say “yes, that's great.”

Use this structure:

### First understand it

“What's the clinical decision you're imagining that would come from that?”

### Then understand the data

“Do we already have the data needed to answer it?”

### Then understand the comparison

“What would you consider the appropriate baseline?”

### Then understand success

“What result would actually convince you this was clinically useful?”

### Finally respond

“That makes sense. I think the first thing I'd want to do is [smallest decisive experiment]. If that works, then it justifies building the more complicated model.”

This makes you sound like a researcher rather than someone collecting project ideas.

---

# 16. If he proposes simply improving the current reconstruction model

Say:

“I can definitely continue improving it. My only concern is that before we invest heavily in another architecture, I think we should test whether our current models have the same regression-to-the-mean problem Presacan identified.

If they do, I think that result should influence the architecture. If they don't, then that's actually a very important finding in itself.”

---

# 17. If he likes the Stanford collaboration idea

Say:

“I can keep the discussion going with Rayyan and get a clearer sense of what they would actually be interested in contributing.

Before making anything formal, I could come back to you and Alex with a one-page outline of the question, what each group brings, and what the resulting study would look like.”

Do not promise authorship arrangements or collaboration scope yourself.

---

# 18. If Chris says everything is paused because of funding

Don't retreat from the conversation.

Say:

“I completely understand. In that case, it would still be useful for me to understand what analyses I can push independently with the data and models we already have, and which opportunities are contingent on funding.”

Then ask:

“Which grants are the main gating items?”

And:

“When do you expect to know more?”

That tells you when to revisit the employment/PhD conversation.

---

# 19. Close the meeting deliberately

Near the end:

“This was really helpful. Let me make sure I've got the priorities right.”

Then summarize aloud:

“So first, I'll ______.”

“Second, we'll look into ______.”

“On the Fitbit side, you'll check / I'll check ______.”

“And for the longer-term PhD/full-time question, it sounds like ______.”

Then:

“I'll send you a short follow-up with the concrete pieces rather than a long summary.”

Finally:

“And once I have the regression-to-the-mean analysis, could we meet again and use that to decide whether the next step is a clinical-validation paper or a new modelling direction?”

---

# What You Want to Leave the Meeting Knowing

By the end, you should have answers to as many of these as possible:

1. What clinical problem Chris actually wants ECG reconstruction to solve.
2. Whether AF recurrence is a sensible use case.
3. Whether paired Fitbit/12-lead acquisition is possible.
4. Whether regression-to-the-mean analysis should be the immediate next experiment.
5. Whether Stanford collaboration interests him.
6. Who he thinks could be strong PhD supervisors/co-supervisors.
7. Whether he sees himself potentially supervising or co-supervising you.
8. What he believes you need to strengthen before PhD applications.
9. What research funding he has or is pursuing.
10. Whether a funded full-time role beginning summer 2027 is realistically possible.
11. What his research program will look like over the next few years.
12. When you should revisit the PhD/employment discussion.

# Information You Should Not Volunteer Yet

Do not frame any option as a fallback.

Avoid:

* “If med school doesn't work out...”
* “If I don't get into a good PhD...”
* “I need a backup job for my MCAT year.”
* “I'm not sure whether I actually want a PhD.”
* “Can you guarantee me funding next year?”

Instead, your position is:

> You are deliberately evaluating several serious long-term training paths, you have a window to invest heavily in research, and you want to understand whether there is a sufficiently strong scientific and mentorship opportunity to justify committing to it.

That is accurate without unnecessarily weakening your negotiating position.