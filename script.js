// Data structures
const drills = {
  earthquake: {
    title: "Earthquake Drill",
    content: `
      <h6>Before:</h6>
      <img src="assets/imgs/before.png" alt="Earthquake Preparation" class="drill-img">
      <ul>
        <li>Secure heavy furniture and objects; anchor tall shelves.</li>
        <li>Store emergency kit with water (3 days), food, torch, radio, meds.</li>
        <li>Plan and practice two evacuation routes and assembly point.</li>
      </ul>
      <h6>During:</h6>
      <img src="assets/imgs/during.png" alt="Earthquake Safety" class="drill-img">
      <ul>
        <li>Drop, Cover and Hold On under a sturdy table or against an interior wall.</li>
        <li>If outside, move to an open area away from structures and power lines.</li>
      </ul>
      <h6>After:</h6>
      <img src="assets/imgs/after.png" alt="After Earthquake" class="drill-img">
      <ul>
        <li>Check for injuries and hazards (gas smell, structural damage).</li>
        <li>Expect aftershocks; evacuate if building is unsafe.</li>
      </ul>
    `,
    image: "assets/imgs/earthquake.jpg",
    buttonClass: "btn-primary"
  },
  tsunami: {
    title: "Tsunami Drill",
    content: `
      <h6>Before:</h6>
      <img src="assets/imgs/tsunamibefore.png" alt="Tsunami Preparation" class="drill-img">
      <ul>
        <li>Know evacuation routes to high ground and nearest shelters.</li>
        <li>Prepare a compact go-bag with essentials and a battery radio.</li>
      </ul>
      <h6>During:</h6>
      <img src="assets/imgs/tsunamiduring.png" alt="Tsunami Safety" class="drill-img">
      <ul>
        <li>If you feel a strong quake or see the sea recede — move inland and uphill immediately.</li>
        <li>Do NOT wait for official alerts if natural warnings are present.</li>
      </ul>
      <h6>After:</h6>
      <img src="assets/imgs/tsunamiafter.png" alt="After Tsunami" class="drill-img">
      <ul>
        <li>Stay away from the coast until authorities say it's safe.</li>
        <li>Report missing persons and follow relief instructions.</li>
      </ul>
    `,
    image: "assets/imgs/tsunami.jpg",
    buttonClass: "btn-info"
  },
  flood: {
    title: "Flood Simulation",
    content: `
      <h6>Before:</h6>
      <img src="assets/imgs/floodbefore.png" alt="Flood Preparation" class="drill-img">
      <ul>
        <li>Move valuables/electronics to higher places and seal documents.</li>
        <li>Prepare emergency supplies and know safe higher-ground locations.</li>
      </ul>
      <h6>During:</h6>
      <img src="assets/imgs/floodduring.png" alt="Flood Safety" class="drill-img">
      <ul>
        <li>Move to higher ground immediately; avoid walking/driving through floodwater.</li>
        <li>Turn off utilities only if it is safe to do so.</li>
      </ul>
      <h6>After:</h6>
      <img src="assets/imgs/floodafter.png" alt="After Flood" class="drill-img">
      <ul>
        <li>Beware of contaminated water; document damage for claims.</li>
        <li>Dry, clean and disinfect items exposed to floodwater.</li>
      </ul>
    `,
    image: "assets/imgs/flood.jpg",
    buttonClass: "btn-primary"
  },
  fire: {
    title: "Forest Fire Drill",
    content: `
      <h6>Before:</h6>
      <img src="assets/imgs/forestbefore.png" alt="Fire Preparation" class="drill-img">
      <ul>
        <li>Create defensible space by clearing leaves and dry brush near buildings.</li>
        <li>Keep an evacuation bag ready including masks for smoke.</li>
      </ul>
      <h6>During:</h6>
      <img src="assets/imgs/orest during.png" alt="Fire Safety" class="drill-img">
      <ul>
        <li>Evacuate immediately when ordered; wear protective clothing and mask.</li>
        <li>Close doors and windows if leaving by vehicle; keep windows closed while escaping smoke.</li>
      </ul>
      <h6>After:</h6>
      <img src="assets/imgs/forestafter.png" alt="After Fire" class="drill-img">
      <ul>
        <li>Return only after authorities confirm; watch for hotspots and structure damage.</li>
        <li>Seek medical attention for smoke inhalation.</li>
      </ul>
    `,
    image: "assets/imgs/forest fire.jpg",
    buttonClass: "btn-danger"
  },
  cyclone: {
    title: "Cyclone Drill",
    content: `
      <h6>Before:</h6>
      <img src="assets/imgs/cyclonebefore.png" alt="Cyclone Preparation" class="drill-img">
      <ul>
        <li>Secure loose outdoor objects; board up windows if advised.</li>
        <li>Stock up on water, food, medicine and charge devices.</li>
      </ul>
      <h6>During:</h6>
      <img src="assets/imgs/cycloneduring.png" alt="Cyclone Safety" class="drill-img">
      <ul>
        <li>Stay indoors in an interior room away from windows.</li>
        <li>Follow official broadcasts; do not travel unless safe.</li>
      </ul>
      <h6>After:</h6>
      <img src="assets/imgs/cycloneafter.png" alt="After Cyclone" class="drill-img">
      <ul>
        <li>Avoid downed power lines; inspect your property for hazards.</li>
        <li>Assist neighbors and report damage to authorities.</li>
      </ul>
    `,
    image: "assets/imgs/cyclone.jpg",
    buttonClass: "btn-warning"
  },
  landslide: {
    title: "Landslide Drill",
    content: `
      <h6>Before:</h6>
      <img src="assets/imgs/landbefore.png" alt="Landslide Preparation" class="drill-img">
      <ul>
        <li>Avoid building near steep slopes; maintain proper drainage around property.</li>
        <li>Prepare evacuation routes uphill and a go-bag.</li>
      </ul>
      <h6>During:</h6>
      <img src="assets/imgs/landduring.png" alt="Landslide Safety" class="drill-img">
      <ul>
        <li>If you hear rumbling or see moving earth, move uphill immediately.</li>
        <li>Help vulnerable people evacuate and keep away from river valleys.</li>
      </ul>
      <h6>After:</h6>
      <img src="assets/imgs/landafter.png" alt="After Landslide" class="drill-img">
      <ul>
        <li>Keep away from debris and report blocked roads or damaged structures.</li>
        <li>Watch for secondary slides and after-effects.</li>
      </ul>
    `,
    image: "assets/imgs/landslide.jpg",
    buttonClass: "btn-secondary"
  }
};

const resourcesData = {
  earthquake: { videos: [ "https://www.youtube.com/embed/BLEPakj1YTY", "https://www.youtube.com/embed/liw3hnAyV8U?si=k4eXH2IV24B5l3vV" ] },
  tsunami: { videos: [ "https://www.youtube.com/embed/7EDflnGzjTY?si=c1pHy3TU9KVsJ_TY", "https://www.youtube.com/embed/KOJdArJCQGI?si=yY3VNIUKTw0TVijx" ] },
  flood: { videos: [ "https://www.youtube.com/embed/cqCMXSOo8qc?si=s0vLs3G_ED9iYlgW", "https://www.youtube.com/embed/rV1iqRD9EKY?si=uTAzNDndIaZSwiae" ] },
  fire: { videos: [ "https://www.youtube.com/embed/Uc9uIZB4xvQ?si=9rSLdrPhiwI7hnWi", "https://www.youtube.com/embed/_bNLtjHG9dM?si=mPg06WLTNCtjWc4X" ] },
  cyclone: { videos: [ "https://www.youtube.com/embed/B9qR2e3xyJo?si=fAhV_Rht9XhxOcIT", "https://www.youtube.com/embed/xHRbnuB9F1I?si=JVVS9PkYd9h_vjLH" ] },
  landslide: { videos: [ "https://www.youtube.com/embed/VcgoZlpn1Y4?si=f5ad818FlJ6IxhPf", "https://www.youtube.com/embed/eSq6_rX_kOc?si=t0WRrpD-vXaG3kMT" ] }
};

// Generate drill cards
function generateDrillCards() {
  const container = document.getElementById('drill-cards');
  for (let key in drills) {
    const drill = drills[key];
    const card = `
      <div class="col-md-4">
        <div class="card shadow-sm h-100">
          <img src="${drill.image}" class="card-img-top" alt="${drill.title}">
          <div class="card-body">
            <h5 class="card-title"><i class="fas fa-${getIcon(key)} me-2"></i>${drill.title}</h5>
            <p class="card-text">${getDescription(key)}</p>
            <a href="#" class="btn ${drill.buttonClass} btn-sm" data-bs-toggle="modal" data-bs-target="#${key}Modal">View Steps</a>
          </div>
        </div>
      </div>
    `;
    container.innerHTML += card;
  }
}

function getIcon(key) {
  const icons = {
    earthquake: 'house-damage',
    tsunami: 'water',
    flood: 'house-flood-water',
    fire: 'fire',
    cyclone: 'wind',
    landslide: 'mountain'
  };
  return icons[key] || 'exclamation-triangle';
}

function getDescription(key) {
  const descriptions = {
    earthquake: 'Practice "Drop, Cover, and Hold On" to stay safe during an earthquake.',
    tsunami: 'Practice safe evacuation to high ground when a tsunami warning is issued.',
    flood: 'Prepare for rising water levels and practice safe evacuation to higher ground.',
    fire: 'Learn evacuation routes and practice fire safety strategies.',
    cyclone: 'Prepare for strong winds and follow safe shelter procedures.',
    landslide: 'Practice quick evacuation and slope safety awareness during landslides.'
  };
  return descriptions[key] || 'Learn important safety procedures.';
}

// Generate modals
function generateModals() {
  const container = document.getElementById("modals-container");
  for (let key in drills) {
    let modal = `
      <div class="modal fade" id="${key}Modal" tabindex="-1">
        <div class="modal-dialog modal-lg modal-dialog-centered">
          <div class="modal-content">
            <div class="modal-header">
              <h5 class="modal-title">${drills[key].title}</h5>
              <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">${drills[key].content}</div>
            <div class="modal-footer">
              <button class="btn btn-outline-primary" onclick="downloadPDF('${key}')">Download PDF</button>
              <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
          </div>
        </div>
      </div>
    `;
    container.innerHTML += modal;
  }
}

// PDF download function
async function downloadPDF(key) {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF("p", "pt", "a4");

  const modalBody = document.querySelector(`#${key}Modal .modal-body`);

  const canvas = await html2canvas(modalBody, { scale: 2 });
  const imgData = canvas.toDataURL("image/png");

  const imgProps = doc.getImageProperties(imgData);
  const pdfWidth = doc.internal.pageSize.getWidth() - 40;
  const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

  doc.addImage(imgData, "PNG", 20, 20, pdfWidth, pdfHeight);
  doc.save(drills[key].title + ".pdf");
}

// Render resource tabs content
function renderResources() {
  const container = document.getElementById('resources-content');
  container.innerHTML = '';

  for (const key of Object.keys(resourcesData)) {
    const res = resourcesData[key];
    const active = key === 'earthquake' ? 'show active' : '';
    const pane = document.createElement('div');
    pane.className = `tab-pane fade ${active}`;
    pane.id = key + 'Res';

    let html = '<div class="row justify-content-center mt-3">';
    res.videos.forEach(video => {
      html += `
        <div class="col-md-6 text-center mb-3">
          <div class="ratio ratio-16x9">
            <iframe src="${video}" title="YouTube video" allowfullscreen></iframe>
          </div>
        </div>`;
    });
    html += '</div>';

    pane.innerHTML = html;
    container.appendChild(pane);
  }
}

// Form validation
function setupFormValidation() {
  const form = document.getElementById('drillForm');
  form.addEventListener('submit', function (event) {
    if (!form.checkValidity()) {
      event.preventDefault()
      event.stopPropagation()
    } else {
      event.preventDefault();
      alert('Thank you for scheduling a drill! We will contact you shortly to confirm the details.');
      form.reset();
      form.classList.remove('was-validated');
    }
    form.classList.add('was-validated')
  }, false);
}

// Initialize everything
function init() {
  generateDrillCards();
  generateModals();
  renderResources();
  setupFormValidation();
}

// Call on load
document.addEventListener('DOMContentLoaded', init);