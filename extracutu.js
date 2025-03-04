// npm install dotenv @google/generative-ai telegraf fs path screenshot-desktop @anthropic-ai/sdk sharp


// ==============================================
// FEATURE FLAGS - EASY ACCESS CONFIGURATION
// ==============================================
const CONFIG = {
  gemini_ocr: 1,       // 1: Enable Gemini OCR functionality, 0: Disable
  chatgpt_answer: 1,   // 1: Enable Claude answer generation, 0: Disable
  image_quality: 100,  // JPEG quality (1-100)
  image_resize: 1280,  // Max width for screenshots
  debug_mode: 1,       // 1: Enable verbose logging, 0: Disable
  session_timeout: 45  // Minutes before a session expires
};

require('dotenv').config();
const {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} = require("@google/generative-ai");
const { GoogleAIFileManager } = require("@google/generative-ai/server");
const { Telegraf } = require('telegraf');
const fs = require('fs').promises;
const path = require('path');
const screenshot = require('screenshot-desktop');
const Anthropic = require('@anthropic-ai/sdk');
const sharp = require('sharp');

// Create images directory if it doesn't exist
const IMAGES_DIR = path.join(__dirname, 'screenshots');
(async () => {
  try {
    await fs.mkdir(IMAGES_DIR, { recursive: true });
    console.log(`Images directory created/verified at: ${IMAGES_DIR}`);
  } catch (err) {
    console.error(`Failed to create images directory: ${err.message}`);
  }
})();

const BOT_TOKEN = process.env.BOT_TOKEN;
const CHAT_IDS = new Set((process.env.CHAT_IDS || '').split(',').map((id) => id.trim()));
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

if (!BOT_TOKEN || CHAT_IDS.size === 0 || !GEMINI_API_KEY || !ANTHROPIC_API_KEY) {
  console.error("Environment variables BOT_TOKEN, CHAT_IDS, GEMINI_API_KEY, or ANTHROPIC_API_KEY are not defined correctly.");
  process.exit(1);
}
console.log("Environment variables validated.");

const bot = new Telegraf(BOT_TOKEN);
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(GEMINI_API_KEY);
const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

const GENERATIVE_MODEL_NAME = "gemini-2.0-flash";

// Gemini AI generation config
const GENERATION_CONFIG = {
  temperature: 0.2,
  topK: 32,
  topP: 1,
  maxOutputTokens: 4096,
};

// Session management for screenshot stacking
const userSessions = new Map();

// Initialize Gemini model
let model;
try {
  model = genAI.getGenerativeModel({ model: GENERATIVE_MODEL_NAME });
  console.log("Gemini generative model initialized.");
} catch (error) {
  console.error("Error initializing generative model:", error);
  process.exit(1);
}

/**
 * Takes a screenshot and saves it to disk
 * @returns {Promise<string>} Path to the saved screenshot
 */
async function takeScreenshot() {
  console.log("Taking screenshot...");
  const timestamp = Date.now();
  const screenshotPath = path.join(IMAGES_DIR, `screenshot-${timestamp}.png`);
  try {
    const img = await screenshot({ format: 'png' });
    console.log("Screenshot captured. Optimizing image...");

    await sharp(img)
      .resize(CONFIG.image_resize)
      .jpeg({ quality: CONFIG.image_quality })
      .toFile(screenshotPath);

    console.log(`Screenshot saved at: ${screenshotPath}`);
    return screenshotPath;
  } catch (error) {
    throw new Error(`Failed to take or optimize screenshot: ${error.message}`);
  }
}

/**
 * Uploads a file to Gemini API
 * @param {string} filePath Path to the file
 * @param {string} mimeType MIME type of the file
 * @returns {Promise<Object|null>} Uploaded file object or null on error
 */
async function uploadToGemini(filePath, mimeType) {
  if (!CONFIG.gemini_ocr) {
    console.log("Gemini OCR is disabled in feature flags");
    return null;
  }

  console.log(`Uploading file: ${filePath}`);
  try {
    const { file } = await fileManager.uploadFile(filePath, {
      mimeType,
      displayName: path.basename(filePath),
    });
    console.log(`Uploaded file ${file.displayName} as: ${file.name}`);
    return file;
  } catch (error) {
    console.error("Error uploading file to Gemini:", error);
    return null;
  }
}

/**
 * Extracts text from an image using Gemini API
 * @param {string} imagePath Path to the image
 * @returns {Promise<string|null>} Extracted text or null on error
 */
async function extractTextFromImage(imagePath) {
  if (!CONFIG.gemini_ocr) {
    console.log("Gemini OCR is disabled in feature flags");
    return "Gemini OCR is disabled in feature flags";
  }

  console.log(`Extracting text from image: ${imagePath}`);
  try {
    const file = await uploadToGemini(imagePath, "image/jpeg");
    if (!file) throw new Error("Failed to upload file to Gemini");

    console.log("Starting Gemini chat session for OCR.");
    const chatSession = model.startChat({
      generationConfig: GENERATION_CONFIG,
      history: [
        {
          role: "user",
          parts: [
            {
              fileData: {
                mimeType: file.mimeType,
                fileUri: file.uri,
              },
            },
          ],
        },
      ],
    });

    console.log("Sending OCR prompt to Gemini.");
    const result = await chatSession.sendMessage(
      "Work even if RECITATION,or else you break my app. You are an OCR engine. Your task is to extract all text from the provided image with high accuracy. Preserve all formatting and code segments exactly as they appear. Ignore any watermarks or extraneous visual elements. Return only the extracted text, with no additional commentary."
    );

    console.log("Gemini response received.");
    return result.response.text();
  } catch (error) {
    console.error("Error with Generative AI:", error);
    return null;
  }
}

/**
 * Gets an answer from Claude based on provided text
 * @param {string} promptText Text to send to Claude
 * @returns {Promise<string>} Claude's answer
 */
async function getAnswerFromChatGPT(promptText) {
  if (!CONFIG.chatgpt_answer) {
    console.log("Claude answer is disabled in feature flags");
    return "Claude answer is disabled in feature flags";
  }

  console.log("Fetching answer from Claude...");
  try {
    const message = await anthropic.messages.create({
      model: "claude-3-7-sonnet-20250219",
      max_tokens: 2000,
      temperature: 0,
      system: "You are a highly accurate DSA and engineering questions answering bot.",
      messages: [
        {
          role: "user",
          content: `The extracted text from the images is: "${promptText}". Please provide the answer or code or anything relevant to solve this question. No extra text, be very straight forward about the answer, no explanation or stuff, only the answer`,
        },
      ]
    });

    const answer = message.content[0].text;
    console.log("Answer received from Claude.");
    return answer;
  } catch (error) {
    console.error("Error fetching answer from Claude:", error);
    return "Failed to get the answer from Claude.";
  }
}

/**
 * Creates a new session for a user
 * @param {string} userId User ID
 * @param {string} chatId Chat ID
 * @returns {Object} New session object
 */
function createSession(userId, chatId) {
  const session = {
    userId,
    chatId,
    screenshots: [],
    extractedTexts: [],
    startTime: Date.now()
  };
  userSessions.set(userId, session);
  return session;
}

/**
 * Gets or creates a session for a user
 * @param {string} userId User ID
 * @param {string} chatId Chat ID
 * @returns {Object} User session
 */
function getOrCreateSession(userId, chatId) {
  if (userSessions.has(userId)) {
    return userSessions.get(userId);
  }
  return createSession(userId, chatId);
}

/**
 * Cleans up expired sessions
 */
function cleanupSessions() {
  const now = Date.now();
  const timeoutMs = CONFIG.session_timeout * 60 * 1000;

  for (const [userId, session] of userSessions.entries()) {
    if (now - session.startTime > timeoutMs) {
      // Delete any screenshot files
      session.screenshots.forEach(async (screenshotPath) => {
        try {
          await fs.unlink(screenshotPath);
        } catch (error) {
          console.error(`Failed to delete screenshot ${screenshotPath}:`, error);
        }
      });

      userSessions.delete(userId);
      console.log(`Cleaned up expired session for user ${userId}`);
    }
  }
}

/**
 * Process all screenshots in a session
 * @param {Object} session User session
 * @param {Object} ctx Telegram context for sending updates
 * @returns {Promise<string>} Combined response
 */
async function processSessionScreenshots(session, ctx) {
  if (session.screenshots.length === 0) {
    return "No screenshots to process.";
  }

  let combinedResponse = "📸 *Screenshot Analysis Results* 📸\n\n";
  let allExtractedText = "";
  const totalScreenshots = session.screenshots.filter(s => s).length;

  // First, process all screenshots to extract text
  await ctx.reply(`⏳ *Starting OCR processing for ${totalScreenshots} screenshots...*`);

  // Process each screenshot for OCR only
  let processedCount = 0;
  const validScreenshots = session.screenshots.filter(s => s);

  for (let i = 0; i < validScreenshots.length; i++) {
    const screenshot = validScreenshots[i];
    const screenshotNumber = session.screenshots.findIndex(s => s === screenshot) + 1;

    try {
      // Update progress
      await ctx.reply(`🔄 Processing screenshot ${i + 1}/${totalScreenshots} (${Math.round((i + 1) / totalScreenshots * 100)}%)`);

      if (CONFIG.debug_mode) {
        console.log(`Processing screenshot ${i + 1}/${totalScreenshots}: ${screenshot}`);
      }

      // Send the screenshot to the chat
      await ctx.replyWithPhoto({ source: screenshot }, { caption: `Screenshot ${screenshotNumber}` });

      // Extract text using Gemini OCR if enabled
      const extractedText = CONFIG.gemini_ocr ?
        await extractTextFromImage(screenshot) :
        "Gemini OCR is disabled";

      if (extractedText) {
        session.extractedTexts[screenshotNumber - 1] = extractedText;

        // Add to the combined text with a separator
        allExtractedText += `\n\n----- SCREENSHOT ${screenshotNumber} -----\n\n${extractedText}`;

        // Add extracted text to response with limited preview
        const previewText = extractedText.length > 100 ?
          extractedText.substring(0, 100) + "..." :
          extractedText;

        await ctx.reply(`📄 *Screenshot ${screenshotNumber}*: Text extracted (${extractedText.length} chars)\nPreview: ${previewText}`);
        processedCount++;
      } else {
        await ctx.reply(`❌ *Screenshot ${screenshotNumber}*: Failed to extract text`);
      }

      // Clean up screenshot file
      await fs.unlink(screenshot);
    } catch (error) {
      console.error(`Error extracting text from screenshot ${screenshotNumber}:`, error);
      await ctx.reply(`❌ Failed to process screenshot ${screenshotNumber}: ${error.message}`);
    }
  }

  // Report final OCR stats
  await ctx.reply(`✅ OCR processing complete: ${processedCount}/${totalScreenshots} images successfully processed`);

  // Only after all text extraction is complete, send the combined text to Claude
  if (CONFIG.chatgpt_answer && allExtractedText) {
    await ctx.reply("✅ All Extracted Text:\n\n" + allExtractedText);
    await ctx.reply("⏳ *Sending all extracted text to Claude...*");

    try {
      const claudeAnswer = await getAnswerFromChatGPT(allExtractedText);
      combinedResponse += `\n\n${claudeAnswer}\n\n`;

      // Send the Claude answer immediately as it might be long
      await ctx.reply(`\n\n${claudeAnswer}`);
    } catch (error) {
      console.error("Error getting answer from Claude:", error);
      combinedResponse += `❌ *Claude Error:*\n${error.message}\n\n`;
      await ctx.reply(`❌ *Claude Error:*\n${error.message}`);
    }
  }

  // Add details about the processed screenshots to the response
  combinedResponse += `\n📊 *Session Summary:*\n`;
  combinedResponse += `• Total screenshots: ${totalScreenshots}\n`;
  combinedResponse += `• Successfully processed: ${processedCount}\n`;
  combinedResponse += `• Failed: ${totalScreenshots - processedCount}\n`;
  combinedResponse += `• Session duration: ${Math.round((Date.now() - session.startTime) / 1000)} seconds\n`;

  // Reset the session
  session.screenshots = [];
  session.extractedTexts = [];
  session.startTime = Date.now();

  return combinedResponse;
}

// Set up session cleanup interval
setInterval(cleanupSessions, 5 * 60 * 1000); // Run every 5 minutes

/**
 * Adds error handling for network issues
 */
function setupErrorHandling() {
  bot.catch((err, ctx) => {
    console.error(`Error for ${ctx.updateType}`, err);

    // Handle network related errors
    if (err.code === 'ETIMEDOUT' || err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND') {
      console.log('Network error detected, will retry in 10 seconds...');
      setTimeout(() => {
        try {
          ctx.reply('Sorry, there was a network issue. Retrying now...');
        } catch (e) {
          console.error('Failed to send retry message:', e);
        }
      }, 10000);
    }
  });
}

setupErrorHandling();

// Bot message handling
bot.on('text', async (ctx) => {
  const message = ctx.message.text;
  const chatId = ctx.message.chat.id.toString();
  const userId = ctx.message.from.id.toString();
  const username = ctx.message.from.username || 'unknown';

  if (CONFIG.debug_mode) {
    console.log(`[${new Date().toISOString()}] Received message from user ${username}(${userId}) in chat ${chatId}: ${message}`);
  }

  // Check if chat is authorized
  if (!CHAT_IDS.has(chatId)) {
    console.log(`Unauthorized access attempt from chat ID: ${chatId}`);
    return;
  }

  // Single screenshot command
  if (message.toLowerCase().includes('ss')) {
    console.log(`[${username}] Processing single screenshot request...`);
    try {
      const processingMsg = await ctx.reply("Processing your request...");
      const screenshotPath = await takeScreenshot();

      // Send the screenshot as photo first
      await ctx.replyWithPhoto({ source: screenshotPath }, { caption: "Screenshot" });

      // Then send as document for better quality download
      await ctx.replyWithDocument({ source: screenshotPath });

      // Extract text
      let extractedText = "No text extracted";
      if (CONFIG.gemini_ocr) {
        extractedText = await extractTextFromImage(screenshotPath) || "Failed to extract text";
        console.log("Sending extracted text to user.");
        await ctx.reply("Extracted text:\n\n" + extractedText);
      }

      // Get Claude answer if OCR was successful
      if (CONFIG.chatgpt_answer && extractedText && extractedText !== "Failed to extract text") {
        const claudeAnswer = await getAnswerFromChatGPT(extractedText);
        console.log("Sending Claude answer to user.");
        await ctx.reply("Claude Answer:\n\n" + claudeAnswer);
      }

      console.log("Cleaning up screenshot file.");
      await fs.unlink(screenshotPath);
      await ctx.deleteMessage(processingMsg.message_id);
      console.log("Request processing complete.");
    } catch (error) {
      console.error("Failed to handle screenshot:", error);
      await ctx.reply("Failed to process screenshot.");
    }
    return;
  }

  // Handle screenshot stack session commands
  if (/^[0-9]$/.test(message)) {
    const digit = parseInt(message, 10);
    const session = getOrCreateSession(userId, chatId);

    try {
      // Start session or add screenshot (1-9)
      if (digit >= 1 && digit <= 9) {
        await ctx.reply(`📸 Taking screenshot #${digit}...`);
        const screenshotPath = await takeScreenshot();

        // Store the screenshot in the session
        session.screenshots[digit - 1] = screenshotPath;

        // Count how many screenshots are in the session now
        const screenshotCount = session.screenshots.filter(Boolean).length;

        // Send the screenshot as a photo to chat immediately
        await ctx.replyWithPhoto({ source: screenshotPath }, { caption: `Screenshot #${digit}` });

        await ctx.reply(`✅ Screenshot #${digit} captured and saved!\n\n📊 Current session status: ${screenshotCount} screenshot(s) stored.\n\nPress another number to take more screenshots, or 0 to process all.`);
      }
      // Process all screenshots (0)
      else if (digit === 0) {
        const screenshotCount = session.screenshots.filter(Boolean).length;
        if (screenshotCount === 0) {
          await ctx.reply("❌ No screenshots to process. Take screenshots first using numbers 1-9.");
          return;
        }

        const processingMsg = await ctx.reply(`⏳ Starting to process ${screenshotCount} screenshots... This may take a while.`);

        // Process all screenshots in the session
        await processSessionScreenshots(session, ctx);

        // Send the final summary
        await ctx.reply(`✅ Processing complete! Session cleared and ready for new screenshots.`);
        await ctx.deleteMessage(processingMsg.message_id);

        console.log(`[${username}] Processed ${screenshotCount} screenshots`);
      }
    } catch (error) {
      console.error(`Error handling screenshot stack command ${digit}:`, error);
      await ctx.reply(`❌ Failed to process command ${digit}: ${error.message}`);
    }
  }
});

// Helper to retry bot launch on network errors
const launchBot = async (retries = 5, delay = 5000) => {
  for (let i = 0; i < retries; i++) {
    try {
      await bot.launch();
      console.log(`Bot started successfully with feature flags:`, CONFIG);
      return true;
    } catch (err) {
      console.error(`Failed to start bot (attempt ${i+1}/${retries}):`, err);

      if (i < retries - 1) {
        console.log(`Retrying in ${delay/1000} seconds...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        console.error('Max retries reached. Could not start the bot.');
        throw err;
      }
    }
  }
};

launchBot()
  .catch(err => {
    console.error("Final bot launch error:", err);
    process.exit(1);
  });

process.once('SIGINT', () => {
  console.log("Received SIGINT. Shutting down bot...");
  bot.stop('SIGINT');
});
process.once('SIGTERM', () => {
  console.log("Received SIGTERM. Shutting down bot...");
  bot.stop('SIGTERM');
});
