#include <WiFi.h>
#include <WebServer.h>
#include <DHT.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// DHT Sensor setup
#define DHTPIN 4
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

// Soil moisture sensor pin
#define SOIL_PIN 34

WebServer server(80);

void handleRoot() {

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  int soilValue = analogRead(SOIL_PIN);

  // Convert soil value to percentage
  int soilPercent = map(soilValue, 4095, 0, 0, 100);

  // Check if DHT sensor failed
  if (isnan(humidity) || isnan(temperature)) {
    server.send(500, "text/plain", "Failed to read from DHT sensor!");
    return;
  }

  String html = R"rawliteral(
  <!DOCTYPE html>
  <html>
  <head>
      <title>ESP32 Smart Agriculture Dashboard</title>
      <meta http-equiv="refresh" content="5">
      <style>
          body {
              font-family: Arial;
              text-align: center;
              background: #f4f4f4;
              margin-top: 50px;
          }

          .card {
              background: white;
              padding: 20px;
              margin: auto;
              width: 300px;
              border-radius: 10px;
              box-shadow: 0 0 10px rgba(0,0,0,0.2);
          }

          h1 {
              color: green;
          }

          p {
              font-size: 20px;
          }
      </style>
  </head>

  <body>
      <div class="card">
          <h1>Smart Agriculture Dashboard</h1>
  )rawliteral";

  html += "<p>🌡 Temperature: " + String(temperature) + " °C</p>";
  html += "<p>💧 Humidity: " + String(humidity) + " %</p>";
  html += "<p>🌱 Soil Moisture: " + String(soilPercent) + " %</p>";

  html += R"rawliteral(
      </div>
  </body>
  </html>
  )rawliteral";

  server.send(200, "text/html", html);
}

void setup() {

  Serial.begin(115200);

  dht.begin();

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);

  server.begin();

  Serial.println("Web Server Started");
}

void loop() {
  server.handleClient();
}
