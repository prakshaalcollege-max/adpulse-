# AdPulse AI: Customer Discovery Survey (24 questions)

**Title shown to respondents:** How UAE online brands measure their advertising

**Introduction:** This 8-minute survey is part of a business analytics project at SP Jain School of Global Management, Dubai. It is anonymous. Please answer only if you make or influence decisions about paid online advertising for a business selling in the UAE.

**Screening:** respondents who answer "We do not run paid ads" at Q5 are thanked and exit.

**Status:** ready to field (Google Forms). No responses have been collected yet, and none are reported in the report.

---

## Section A: Business profile

1. **What is your role?** Founder / owner · Head of marketing · Performance / digital marketer · Agency (managing clients' ads) · Other
2. **What does your business (or main client) mainly sell?** Fragrance & beauty · Fashion & accessories · Electronics · Home & furniture · Food & health · Baby & kids · Services · Other
3. **Where do you sell online? (select all)** Own website (Shopify / WooCommerce / Salla / other) · Amazon.ae · Noon · Instagram / WhatsApp orders · Other
4. **What is your average monthly spend on paid online advertising?** Under AED 10k · AED 10-40k · AED 40-100k · AED 100-250k · Over AED 250k · Prefer not to say
5. **Which paid channels do you use? (select all)** Meta (Facebook / Instagram) · Google Search / Shopping · TikTok · Snapchat · YouTube · Influencers · Other · We do not run paid ads

## Section B: Current practice

6. **How often do you review ad results and move budget between channels?** Daily · Weekly · Monthly · Only around big campaigns · Rarely
7. **Which source do you mainly trust to judge ad performance?** Each platform's own dashboard · Google Analytics · Store backend (e.g., Shopify orders) · A spreadsheet combining sources · Agency report · Not sure
8. **How many hours a week does your team spend compiling ad reports?** Under 1 · 1-3 · 3-6 · 6-10 · Over 10
9. **Do you calculate profit after product cost for each campaign (not only ROAS)?** Always · Sometimes · Never · Not sure what this means
10. **Do you track how many customers from each channel buy again?** Yes, regularly · Partly · No
11. **Have you ever run a holdout or geo test (turning ads off for a group to measure the real effect)?** Yes · No · I don't know what this is

## Section C: Pain points (1 = strongly disagree, 5 = strongly agree)

12. "When I add up the sales reported by each ad platform, the total is higher than my real sales." 1-5 · Don't know
13. "I am confident I know which channel gives my business the best return." 1-5
14. "During Ramadan, White Friday and other peak seasons, I am unsure whether extra ad spend pays off." 1-5
15. "I find it hard to decide how to split my budget between channels." 1-5
16. **What is your biggest frustration with measuring your advertising today?** Open text
17. **Roughly what share of your ad budget do you think is wasted?** 0-10% · 10-25% · 25-50% · Over 50% · Don't know

## Section D: Solution fit

18. **Rank these features from most (1) to least (6) useful:** See real vs claimed revenue by channel · Predict whether a campaign will make a profit · Recommend how to split my budget · Show which channels bring repeat customers · Test "what-if" scenarios · Weekly summary in Arabic and English
19. **Would you connect your ad accounts and store data to a tool like this?** Yes · Yes, if data stays in the UAE and access is read-only · Only by uploading CSV files · No
20. **What would stop you from using such a tool? (select up to 2)** Data privacy · Price · Set-up time · Not sure the recommendations are accurate · My team lacks time / skills · Nothing

## Section E: Willingness to pay (Van Westendorp, AED per month)

Shown after a one-paragraph description of AdPulse.

21. At what monthly price would this tool be **so cheap** that you would doubt its quality? AED ____
22. At what monthly price would it be **a bargain**, a great buy for the money? AED ____
23. At what monthly price would it start to feel **expensive**, but you would still consider it? AED ____
24. At what monthly price would it be **too expensive** to consider? AED ____

Optional after Q24: "Would you like a free Truth Check of your own ad data? Leave an email." (stored separately from survey answers)

---

## Hypotheses and analysis plan

| # | Hypothesis | Questions | Test |
|---|---|---|---|
| H1 | Most UAE online brands rely mainly on platform dashboards | Q7 | One-sample proportion test vs 50% |
| H2 | Marketers perceive platform-reported sales exceed real sales | Q12 | One-sample t-test vs neutral (3) |
| H3 | Brands spending > AED 40k/month find budget splitting harder | Q4, Q15 | Mann-Whitney U |
| H4 | Acceptable price range includes AED 590-1,490/month | Q21-Q24 | Van Westendorp price sensitivity meter |
| H5 | Data-sharing concerns are the main adoption barrier | Q19, Q20 | Frequency ranking; chi-square by spend band |

**Sample:** 40-60 complete responses (at least 15 from brands spending over AED 40k/month), recruited through LinkedIn, Dubai founder and SME communities, Shopify/Salla merchant groups and the SP Jain network. Purposive and snowball sampling, so results describe respondents rather than the whole population.
