async function fetchJSON(url) {
  const res = await fetch(url);
  return await res.json();
}

async function render() {
  const success = await fetchJSON('/api/monitoring/success');
  const bottlenecks = await fetchJSON('/api/monitoring/bottlenecks');
  const versions = await fetchJSON('/api/monitoring/blueprint_versions');

  new Chart(document.getElementById('successChart'), {
    type: 'doughnut',
    data: {
      labels: ['Success', 'Failure'],
      datasets: [{
        data: [success.successes, success.failures],
        backgroundColor: ['#4caf50', '#f44336']
      }]
    }
  });

  new Chart(document.getElementById('bottleneckChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(bottlenecks),
      datasets: [{
        data: Object.values(bottlenecks),
        backgroundColor: '#2196f3'
      }]
    }
  });

  new Chart(document.getElementById('blueprintChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(versions),
      datasets: [{
        data: Object.values(versions),
        backgroundColor: '#9c27b0'
      }]
    }
  });
}

render();
