document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  const btn = document.getElementById('analyzeBtn');
  const loader = document.getElementById('loader');
  const resultDiv = document.getElementById('result');

  btn.style.display = 'none';
  loader.style.display = 'block';
  resultDiv.style.display = 'none';

  try {
    // Step 1: Execute script in tab to get the rendered HTML
    const injection = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.documentElement.outerHTML
    });
    
    const htmlContent = injection[0].result;

    // Step 2: Send HTML to the new backend endpoint
    const response = await fetch('http://localhost:8000/api/analyze-html', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        url: tab.url,
        html: htmlContent 
      })
    });

    const data = await response.json();
    loader.style.display = 'none';
    resultDiv.style.display = 'block';

    if (data.results && data.results.length > 0) {
      const sentiments = data.results.map(r => r.sentiment);
      const counts = sentiments.reduce((acc, s) => {
        acc[s] = (acc[s] || 0) + 1;
        return acc;
      }, {});

      const dominant = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
      
      const label = document.getElementById('sentimentLabel');
      label.innerText = dominant;
      label.className = `sentiment-${dominant}`;
      
      document.getElementById('summary').innerText = `Analyzed ${data.results.length} comments from this page. Majority sentiment is ${dominant.toLowerCase()}.`;
    } else {
      document.getElementById('summary').innerText = "No comments found in the current view. Try scrolling down to load them first!";
    }
  } catch (error) {
    console.error(error);
    loader.style.display = 'none';
    btn.style.display = 'block';
    alert("Error: Make sure the Sentiment AI Backend is running and you have loaded the extension correctly.");
  }
});
