import fs from 'fs';

console.log("Hello, world!");
fs.readFile('package.json', 'utf8', (err, data) => {
  if (err) {
    console.error("Error reading file:", err);
    return;
  }
  console.log("File content:", data);
  const jsonData = JSON.parse(data);
  console.log("Parsed JSON:", jsonData);
  console.log("Name:", jsonData.name);
});