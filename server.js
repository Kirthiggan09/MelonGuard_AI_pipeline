import http from "http";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";

dotenv.config();

const PORT = 8081;
const MIME_TYPES = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".json": "application/json",
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml"
};

const server = http.createServer((req, res) => {
  // Support local API proxy for Vercel compatibility
  if (req.url.startsWith("/api/chat") && req.method === "POST") {
    let bodyStr = "";
    req.on("data", chunk => { bodyStr += chunk; });
    req.on("end", async () => {
      dotenv.config({ override: true });
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey || apiKey.trim() === "" || apiKey === "YOUR_GEMINI_API_KEY") {
        let userPrompt = "";
        try {
          const parsed = JSON.parse(bodyStr);
          if (parsed.messages && parsed.messages.length > 0) {
            const lastMsg = parsed.messages[parsed.messages.length - 1];
            userPrompt = Array.isArray(lastMsg.content) 
              ? lastMsg.content.map(c => c.text || "").join(" ") 
              : (lastMsg.content || "");
          }
        } catch (e) {}

        const fallbackAnswer = generateFallbackResponse(userPrompt);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          choices: [
            {
              message: {
                role: "assistant",
                content: fallbackAnswer
              }
            }
          ]
        }));
        return;
      }

      try {
        const response = await fetch("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey.trim()}`
          },
          body: bodyStr
        });
        const data = await response.json();
        res.writeHead(response.status, { "Content-Type": "application/json" });
        res.end(JSON.stringify(data));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: err.message } }));
      }
    });
    return;
  }

  // Normalize request URL path
  let filePath = "." + req.url.split("?")[0];
  if (filePath === "./") {
    filePath = "./index.html";
  }

  const extname = String(path.extname(filePath)).toLowerCase();
  const contentType = MIME_TYPES[extname] || "application/octet-stream";

  fs.readFile(filePath, (error, content) => {
    if (error) {
      if (error.code === "ENOENT") {
        res.writeHead(404, { "Content-Type": "text/html" });
        res.end("<h1>404 Not Found</h1>", "utf-8");
      } else {
        res.writeHead(500);
        res.end(`Sorry, check with the site admin for error: ${error.code} ..\n`);
      }
    } else {
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content, "utf-8");
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});

function generateFallbackResponse(userPrompt) {
  const promptLower = (userPrompt || "").toLowerCase();
  
  if (promptLower.includes("powdery") || promptLower.includes("tepung")) {
    return `### 🍃 Diagnosis: Powdery Mildew (Embun Tepung)
**Punca (Pathogen):** *Golovinomyces cichoracearum* / *Podosphaera xanthii*

**Simptom Visual:**
- Debu putih berkeping halus seperti tepung pada permukaan atas dan bawah daun.
- Daun terjejas menjadi kekuningan, kering, dan akhirnya gugur awal.

**Syor Kawalan Kimia & Biologi:**
1. **Racun Kulat Kimia:**
   - **Azoxystrobin + Difenoconazole** (Contoh: Amistar Top) — 0.5 mL / Liter air.
   - **Myclobutanil** / **Triadimenol** — Spray secara berselang 7-10 hari.
2. **Kawalan Biokawalan (Biocontrol):**
   - Agens biokawalan *Pseudomonas aeruginosa* (PIRG 78%) terbukti mengurangkan penyakit.
3. **Penyembuhan & Keselamatan:**
   - Tempoh dilarang kutip hasil (PHI): 7 hari.
   - Sembur pada waktu awal pagi atau lewat petang.

*Nota: Sila rujuk Pegawai Pertanian Daerah (Jabatan Pertanian Malaysia) untuk pengesahan lanjut.*`;
  }
  
  if (promptLower.includes("downy") || promptLower.includes("perosak")) {
    return `### 🌿 Diagnosis: Downy Mildew (Embun Perosak)
**Punca (Pathogen):** *Pseudoperonospora cubensis*

**Simptom Visual:**
- Bintik-bintik bersiku (angular spots) kekuningan dibatasi urat daun.
- Lapisan spora berwarna keabu-abuan / keunguan pada bawah daun.

**Syor Kawalan Kimia:**
1. **Racun Kulat:**
   - **Metalaxyl + Mancozeb** (Contoh: Ridomil Gold) — 2.0 g / Liter air.
   - **Dimethomorph** atau **Fosetyl-aluminium** — Sembur selang 5-7 hari.
2. **Pengurusan Kebun:**
   - Tingkatkan pengudaraan dan kurangkan kelembapan dalam rumah hijau.
   - PHI: 7-14 hari mengikut label racun.

*Nota: Sila rujuk Pegawai Pertanian Daerah untuk pengesahan.*`;
  }

  if (promptLower.includes("miner") || promptLower.includes("borer") || promptLower.includes("ulat") || promptLower.includes("pelombong")) {
    return `### 🐛 Diagnosis: Leaf Miner / Borer (Ulat Pelombong Daun)
**Perosak:** *Liriomyza sativae* / *Liriomyza trifolii*

**Simptom Visual:**
- Garisan berselok-belok (squiggly white/light green trails) pada permukaan daun.
- Fotosintesis terganggu, daun menjadi layu dan perang.

**Syor Kawalan Kimia:**
1. **Racun Serangga Syorkan:**
   - **Abamectin** (Contoh: Dynamec 1.8EC) — 0.5 mL / Liter air.
   - **Cyromazine** / **Spinetoram** — Berkesan membunuh larva di dalam tisu daun.
2. **Kawalan Fizikal:**
   - Pasang pelekat kuning (yellow sticky traps) di persekitaran rumah hijau.
   - PHI: 3-7 hari.

*Nota: Sila rujuk Pegawai Pertanian Daerah untuk pengesahan.*`;
  }

  return `### 🍈 MelonGuard AI Assistant

Saya MelonGuard AI, penasihat khusus bagi tanaman tembikai susu (rockmelon). Saya boleh membantu anda mengenai **3 isu utama**:
1. 🍃 **Powdery Mildew (Embun Tepung)**
2. 🌿 **Downy Mildew (Embun Perosak)**
3. 🐛 **Leaf Miner / Ulat Pelombong Daun**

*Petunjuk:* Untuk menggunakan model LLM secara langsung (Google Gemini / OpenAI API), masukkan \`GEMINI_API_KEY\` anda di dalam fail \`.env\` atau \`config.js\`.`;
}
