// gen-eval generates eval/testcases.json with 1000 customer-support eval cases.
// Run: go run ./cmd/gen-eval
package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// ── Data model ────────────────────────────────────────────────────────────────

type testCase struct {
	ID          string    `json:"id"`
	Category    string    `json:"category"`
	Description string    `json:"description"`
	Stored      stored    `json:"stored"`
	Query       queryCase `json:"query"`
	Expected    expected  `json:"expected"`
}

type stored struct {
	CacheKey string          `json:"cache_key"`
	Request  request         `json:"request"`
	Response json.RawMessage `json:"response"`
}

type queryCase struct {
	CacheKey  string  `json:"cache_key"`
	Request   request `json:"request"`
	Threshold float64 `json:"threshold,omitempty"`
	CacheType string  `json:"cache_type,omitempty"`
}

type request struct {
	Type     string    `json:"type"`
	Provider string    `json:"provider"`
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
	Params   params    `json:"params"`
}

type message struct {
	Role    string  `json:"role"`
	Content content `json:"content"`
}

type content struct {
	Text string `json:"text"`
}

type params struct {
	Temperature float64 `json:"temperature"`
}

type expected struct {
	Outcome string `json:"outcome"`
	Note    string `json:"note"`
}

// ── Seed data — customer support Q&A pairs ────────────────────────────────────

// fact holds a canonical customer query, a semantically equivalent paraphrase,
// the agent's cached response, and a different query in the same domain (for miss cases).
type fact struct {
	q          string // canonical customer query (stored)
	paraphrase string // reworded — same intent (semantic_hit query)
	answer     string // agent response
	falseQ     string // different intent, same domain (semantic_miss query)
	domain     string
}

var facts = []fact{

	// ── Order management ──────────────────────────────────────────────────────
	{
		"Where is my order?",
		"What is the current status of my order?",
		"I can see your order is currently in transit and expected to arrive within 2–3 business days. You can track it in real time using the link sent to your email.",
		"How do I cancel my order?",
		"order_management",
	},
	{
		"How do I track my shipment?",
		"Can you tell me how to track my delivery?",
		"You can track your shipment by clicking the tracking link in your shipping confirmation email, or by entering your order number on our website under 'My Orders'.",
		"Why is my order delayed?",
		"order_management",
	},
	{
		"My order hasn't arrived yet.",
		"I still haven't received my order.",
		"I'm sorry to hear that! Your order shows as in transit. If it doesn't arrive within 2 more business days, we'll initiate a replacement or full refund at no cost to you.",
		"I received the wrong item.",
		"order_management",
	},
	{
		"How do I cancel my order?",
		"I want to cancel my purchase.",
		"You can cancel your order within 1 hour of placing it by going to 'My Orders' and selecting 'Cancel'. After that window, please contact us and we'll do our best to help.",
		"How do I modify my order?",
		"order_management",
	},
	{
		"Can I change my order after placing it?",
		"Is it possible to modify my order?",
		"Orders can be modified within 30 minutes of purchase. Please contact support immediately and we'll update the details before it ships.",
		"Can I cancel my order?",
		"order_management",
	},
	{
		"I received the wrong item.",
		"The item I received is not what I ordered.",
		"We sincerely apologize for the mix-up! Please keep the incorrect item — we'll ship the correct one to you today at no charge and cover return shipping.",
		"My item arrived damaged.",
		"order_management",
	},
	{
		"My item arrived damaged.",
		"The product I received is broken.",
		"We're so sorry your item arrived damaged. Please send us a photo and we'll send a replacement immediately or issue a full refund — your choice.",
		"I received the wrong item.",
		"order_management",
	},
	{
		"Why is my order delayed?",
		"My order is taking longer than expected, why?",
		"We apologize for the delay. It appears your order has been held at a carrier facility. We've flagged this and expect it to resume transit within 24 hours.",
		"How do I cancel my order?",
		"order_management",
	},
	{
		"I never received my order confirmation email.",
		"I didn't get a confirmation email for my order.",
		"The confirmation email may have landed in your spam folder. If you don't find it there, we can resend it — just verify the email address on your account.",
		"How do I update my email address?",
		"order_management",
	},
	{
		"Can I place an order without an account?",
		"Do I need an account to buy something?",
		"Yes! You can checkout as a guest without creating an account. Simply select 'Guest Checkout' on the payment page.",
		"How do I create an account?",
		"order_management",
	},

	// ── Returns & refunds ─────────────────────────────────────────────────────
	{
		"What is your return policy?",
		"How do I return a product?",
		"We accept returns within 30 days of delivery. Items must be unused and in original packaging. Visit 'My Orders', select the item, and click 'Start a Return'.",
		"How long does a refund take?",
		"returns",
	},
	{
		"How do I initiate a return?",
		"What is the process for returning an item?",
		"To start a return, go to 'My Orders', find the order, and click 'Return Item'. Select your reason and we'll email you a prepaid return label within 24 hours.",
		"What is your return policy?",
		"returns",
	},
	{
		"How long does a refund take?",
		"When will I get my money back?",
		"Refunds are processed within 3–5 business days after we receive your returned item. Your bank may take an additional 2–5 days to post the credit.",
		"How do I initiate a return?",
		"returns",
	},
	{
		"Can I exchange an item instead of returning it?",
		"Is it possible to swap an item for a different one?",
		"Yes! We offer direct exchanges for the same item in a different size or color. Start a return and select 'Exchange' as your reason.",
		"Can I return an item without a receipt?",
		"returns",
	},
	{
		"Can I return an item without the original packaging?",
		"Do I need the original box to make a return?",
		"We prefer items to be returned in original packaging, but it's not always required. Please contact support and we'll review your case individually.",
		"Can I exchange an item?",
		"returns",
	},
	{
		"I returned my item but haven't received a refund.",
		"My return was delivered but I haven't gotten my refund yet.",
		"We show your return was received 2 days ago and is being processed. Refunds typically take 3–5 business days — you should see it by end of week.",
		"How do I track my return shipment?",
		"returns",
	},
	{
		"Do you offer free returns?",
		"Is return shipping free?",
		"Yes — we provide a prepaid return label for all returns within our 30-day window. There is no cost to you.",
		"What is your return policy?",
		"returns",
	},
	{
		"I want a refund, not a replacement.",
		"Can I get my money back instead of a replacement?",
		"Absolutely. When starting your return, select 'Refund' as your preference and we'll process it once we receive the item.",
		"How long does a refund take?",
		"returns",
	},
	{
		"How do I return a gift?",
		"I received a gift I'd like to return.",
		"Gift returns are welcome! We'll issue store credit for the item's current value. You don't need the original receipt — just the order number from the packing slip.",
		"What is your return policy?",
		"returns",
	},
	{
		"What items cannot be returned?",
		"Are there any non-returnable products?",
		"Items marked 'Final Sale', digital downloads, and opened personal hygiene products cannot be returned. All other items are eligible within 30 days.",
		"What is the return policy?",
		"returns",
	},

	// ── Shipping & delivery ───────────────────────────────────────────────────
	{
		"What are your shipping options?",
		"What delivery methods do you offer?",
		"We offer Standard (5–7 days), Expedited (2–3 days), and Overnight shipping. Free standard shipping is available on orders over $50.",
		"How long does shipping take?",
		"shipping",
	},
	{
		"How long does shipping take?",
		"What is the estimated delivery time?",
		"Standard shipping takes 5–7 business days. Expedited takes 2–3 days. Overnight orders placed before 2 PM ship the same day.",
		"What are your shipping options?",
		"shipping",
	},
	{
		"Do you offer free shipping?",
		"Is free delivery available?",
		"Yes! We offer free standard shipping on all orders over $50. Subscribe to our newsletter for occasional free shipping promotions.",
		"How much does shipping cost?",
		"shipping",
	},
	{
		"How much does shipping cost?",
		"What are the delivery fees?",
		"Standard shipping is $4.99, Expedited is $12.99, and Overnight is $24.99. Orders over $50 qualify for free standard shipping.",
		"Do you ship internationally?",
		"shipping",
	},
	{
		"Do you ship internationally?",
		"Can I get my order delivered outside the US?",
		"Yes, we ship to over 50 countries. International shipping rates and delivery times vary by destination — enter your address at checkout to see options.",
		"How much does international shipping cost?",
		"shipping",
	},
	{
		"Can I change my delivery address?",
		"I need to update my shipping address.",
		"If your order hasn't shipped yet, we can update the address. Please contact us immediately with your order number and new address.",
		"Can I cancel my order?",
		"shipping",
	},
	{
		"What happens if I'm not home for delivery?",
		"What if I miss my delivery?",
		"The carrier will typically leave the package at your door or with a neighbour. If a signature is required, they'll leave a notice and attempt redelivery the next business day.",
		"How do I redeliver my package?",
		"shipping",
	},
	{
		"My package says delivered but I didn't receive it.",
		"The tracking shows delivered but the package isn't here.",
		"We're sorry! Please check with neighbours and any secure areas first. If it's still missing after 24 hours, contact us and we'll file a claim and send a replacement.",
		"My order hasn't arrived yet.",
		"shipping",
	},
	{
		"Can I pick up my order in store?",
		"Is in-store pickup available?",
		"Yes! Select 'In-Store Pickup' at checkout and choose your nearest location. You'll receive an email when your order is ready — usually within 2 hours.",
		"What are your shipping options?",
		"shipping",
	},
	{
		"How do I schedule a delivery?",
		"Can I choose my delivery date?",
		"Scheduled delivery is available for large items. During checkout, select 'Schedule Delivery' to pick a date and time window that works for you.",
		"What are your shipping options?",
		"shipping",
	},

	// ── Account management ────────────────────────────────────────────────────
	{
		"How do I reset my password?",
		"I forgot my password, how do I recover it?",
		"Click 'Forgot Password' on the login page, enter your email address, and we'll send a password reset link valid for 15 minutes.",
		"How do I change my email address?",
		"account",
	},
	{
		"How do I update my email address?",
		"I want to change the email on my account.",
		"Go to Account Settings > Personal Information > Email Address. Enter your new email and we'll send a verification link to confirm the change.",
		"How do I reset my password?",
		"account",
	},
	{
		"How do I close my account?",
		"I want to delete my account permanently.",
		"We're sorry to see you go. To close your account, go to Account Settings > Privacy > Delete Account. Note: this cannot be undone and all data will be removed.",
		"How do I cancel my subscription?",
		"account",
	},
	{
		"How do I update my billing information?",
		"I need to change the credit card on my account.",
		"Go to Account Settings > Payment Methods. You can add a new card or update existing details there. Changes take effect immediately.",
		"How do I view my order history?",
		"account",
	},
	{
		"How do I add a new shipping address?",
		"I need to save a new delivery address.",
		"Go to Account Settings > Addresses > Add New Address. You can save multiple addresses and select a default for future orders.",
		"How do I remove a saved address?",
		"account",
	},
	{
		"How do I view my order history?",
		"Where can I see my past orders?",
		"Log in and go to 'My Orders' from the top navigation or account menu. All orders — including cancelled ones — are listed there.",
		"How do I track my current order?",
		"account",
	},
	{
		"I can't log into my account.",
		"I'm having trouble signing into my account.",
		"I'm sorry you're having trouble. Please try resetting your password first. If the problem persists, check that cookies are enabled or try a different browser.",
		"My account was hacked.",
		"account",
	},
	{
		"My account has been hacked.",
		"Someone accessed my account without permission.",
		"Please immediately change your password and enable two-factor authentication. We've flagged your account for review and will contact you within 2 hours.",
		"I can't log into my account.",
		"account",
	},
	{
		"How do I enable two-factor authentication?",
		"Can I add 2FA to my account?",
		"Go to Account Settings > Security > Two-Factor Authentication and follow the setup steps. We support authenticator apps and SMS verification.",
		"How do I reset my password?",
		"account",
	},
	{
		"How do I create an account?",
		"I want to sign up for an account.",
		"Click 'Sign Up' on our homepage, enter your name, email, and a password. You'll receive a verification email to activate your account.",
		"How do I log into my account?",
		"account",
	},

	// ── Payment & billing ─────────────────────────────────────────────────────
	{
		"My payment was declined.",
		"My credit card was rejected at checkout.",
		"Payment declines can happen for several reasons — incorrect card details, insufficient funds, or bank security flags. Please double-check your details or try a different payment method.",
		"How do I update my payment method?",
		"payment",
	},
	{
		"What payment methods do you accept?",
		"Which payment options are available?",
		"We accept Visa, Mastercard, American Express, PayPal, Apple Pay, Google Pay, and gift cards. Buy Now Pay Later is available via Klarna.",
		"Do you accept cryptocurrency?",
		"payment",
	},
	{
		"I was charged twice for the same order.",
		"I see a duplicate charge on my account.",
		"We're sorry about the double charge! One of the transactions is a pending authorisation that will automatically drop off within 3–5 business days. If both post, contact us for an immediate refund.",
		"I was charged the wrong amount.",
		"payment",
	},
	{
		"How do I apply a promo code?",
		"Where do I enter a discount code?",
		"Enter your promo code in the 'Coupon Code' field on the checkout page before completing your order. Click 'Apply' to see the discount reflected in your total.",
		"My promo code isn't working.",
		"payment",
	},
	{
		"My promo code isn't working.",
		"The discount code I have doesn't seem to be valid.",
		"Please check that the code hasn't expired and that your order meets the minimum requirements. Some codes are single-use only. Contact us and we'll verify it for you.",
		"How do I apply a promo code?",
		"payment",
	},
	{
		"How do I get an invoice for my order?",
		"Can I get a receipt or invoice for my purchase?",
		"You can download a PDF invoice for any order from 'My Orders' — click the order and select 'Download Invoice'. We can also email it on request.",
		"How do I get a VAT receipt?",
		"payment",
	},
	{
		"I was charged the wrong amount.",
		"The amount billed to me doesn't match what I expected.",
		"I'm sorry for the billing error. Please share your order number and the amount you were charged. We'll investigate and correct it within 24 hours.",
		"I was charged twice.",
		"payment",
	},
	{
		"Do you offer instalment payments?",
		"Can I pay in instalments?",
		"Yes! We offer Buy Now Pay Later through Klarna and Afterpay. Select your preferred option at checkout and follow the steps to split your payment.",
		"What payment methods do you accept?",
		"payment",
	},
	{
		"How do I use a gift card?",
		"I have a gift card — how do I redeem it?",
		"Enter your gift card number and PIN in the 'Gift Card' field at checkout. Any remaining balance will stay on the card for future use.",
		"How do I apply a promo code?",
		"payment",
	},
	{
		"Can I get a price match?",
		"Do you match competitors' prices?",
		"Yes! If you find the same item cheaper at a major retailer within 7 days of purchase, contact us with proof and we'll match the price or refund the difference.",
		"Can I use a coupon on a sale item?",
		"payment",
	},

	// ── Subscription management ───────────────────────────────────────────────
	{
		"How do I cancel my subscription?",
		"I want to stop my subscription.",
		"Go to Account Settings > Subscriptions > Manage, then select 'Cancel Subscription'. Your access continues until the end of the current billing period.",
		"How do I pause my subscription?",
		"subscription",
	},
	{
		"How do I upgrade my subscription plan?",
		"I want to switch to a higher tier plan.",
		"Go to Account Settings > Subscriptions > Change Plan. Upgrades take effect immediately and you'll only be charged the prorated difference.",
		"How do I downgrade my plan?",
		"subscription",
	},
	{
		"How do I downgrade my plan?",
		"I want to switch to a cheaper plan.",
		"Go to Account Settings > Subscriptions > Change Plan and select the lower tier. The change takes effect at the start of your next billing cycle.",
		"How do I cancel my subscription?",
		"subscription",
	},
	{
		"When is my next billing date?",
		"When will I be charged next for my subscription?",
		"Your subscription renews on the 15th of each month. You can view your exact renewal date in Account Settings > Subscriptions.",
		"How do I cancel my subscription?",
		"subscription",
	},
	{
		"How do I pause my subscription?",
		"Can I put my subscription on hold temporarily?",
		"Yes! Go to Account Settings > Subscriptions > Pause. You can pause for 1, 2, or 3 months. Your billing and access resume automatically at the end of the pause.",
		"How do I cancel my subscription?",
		"subscription",
	},
	{
		"What is included in the premium plan?",
		"What features come with the premium subscription?",
		"The Premium plan includes unlimited access, priority customer support, early sale access, free expedited shipping on all orders, and exclusive member discounts.",
		"What is included in the basic plan?",
		"subscription",
	},
	{
		"How do I get a refund for my subscription?",
		"I was charged for a subscription I didn't want — can I get a refund?",
		"Subscription charges are generally non-refundable, but we review cases individually. If you cancelled and were still charged in error, we'll refund you immediately.",
		"How do I cancel my subscription?",
		"subscription",
	},
	{
		"Can I share my subscription with family?",
		"Does my subscription work for multiple users?",
		"The Family Plan supports up to 5 users. Individual plans are for single-user use. Upgrade to Family in Account Settings > Subscriptions.",
		"What is included in the premium plan?",
		"subscription",
	},
	{
		"What happens to my data if I cancel?",
		"Will I lose my data when I cancel my subscription?",
		"Your data is retained for 90 days after cancellation. You can export it during that time. After 90 days, data is permanently deleted.",
		"How do I close my account?",
		"subscription",
	},
	{
		"How do I reactivate my subscription?",
		"I want to restart my cancelled subscription.",
		"Go to Account Settings > Subscriptions > Reactivate. Your previous plan and settings will be restored and you'll be billed on your original cycle.",
		"How do I cancel my subscription?",
		"subscription",
	},

	// ── Technical support ─────────────────────────────────────────────────────
	{
		"The app is not working.",
		"The application keeps crashing on my phone.",
		"I'm sorry for the trouble! Please try force-closing the app and restarting it. If the issue persists, clear the app cache or reinstall the latest version.",
		"The website is not loading.",
		"technical",
	},
	{
		"The website is not loading.",
		"Your website seems to be down.",
		"We're sorry for the disruption. Our team is aware and working to resolve it. You can check our status page for live updates. Try again in a few minutes.",
		"The app is not working.",
		"technical",
	},
	{
		"I'm not receiving email notifications.",
		"Your emails are not reaching my inbox.",
		"Please check your spam folder first. If emails are going to spam, mark them as 'Not Spam'. Also verify your email address is correct in Account Settings.",
		"How do I unsubscribe from marketing emails?",
		"technical",
	},
	{
		"How do I unsubscribe from marketing emails?",
		"I want to stop receiving promotional emails.",
		"Click 'Unsubscribe' at the bottom of any marketing email, or go to Account Settings > Notifications and toggle off 'Promotional Emails'.",
		"I'm not receiving order confirmation emails.",
		"technical",
	},
	{
		"The checkout page is not working.",
		"I can't complete my purchase — the checkout is broken.",
		"Please try a different browser or clear your cookies and cache. If you're using a VPN, try disabling it. If the issue persists, contact us and we'll place the order for you.",
		"My payment was declined.",
		"technical",
	},
	{
		"I can't upload a photo to my review.",
		"The photo upload in the review section isn't working.",
		"Photo uploads support JPG and PNG files up to 10 MB. Please check your file type and size. If it still fails, try from a desktop browser.",
		"How do I leave a review?",
		"technical",
	},
	{
		"The tracking page shows no information.",
		"My tracking link isn't showing any updates.",
		"Tracking information can take up to 24 hours to appear after your order ships. If it's been longer, the carrier may have a delay. Contact us and we'll investigate.",
		"Where is my order?",
		"technical",
	},
	{
		"I can't find the item I'm looking for.",
		"A product I want doesn't seem to appear in search.",
		"Try searching with different keywords or browse by category. If the item is out of stock, sign up for 'Back in Stock' alerts on the product page.",
		"When will this item be back in stock?",
		"technical",
	},
	{
		"How do I update the app?",
		"The app needs an update — how do I do it?",
		"On iOS, open the App Store, go to your profile, and look for our app under 'Updates'. On Android, open the Play Store and tap 'Update' next to our app.",
		"The app is not working.",
		"technical",
	},
	{
		"Can I use the service on multiple devices?",
		"Does my account work on more than one device?",
		"Yes! Your account works on up to 3 devices simultaneously. Simply log in with the same credentials on each device.",
		"Can I share my subscription?",
		"technical",
	},

	// ── Product questions ─────────────────────────────────────────────────────
	{
		"Is this product available in my size?",
		"Do you have this item in a large size?",
		"Size availability varies by product. Please use the size selector on the product page to check availability. You can sign up for in-stock alerts if your size is unavailable.",
		"What sizes do you carry?",
		"product",
	},
	{
		"When will this item be back in stock?",
		"Is this out-of-stock product available soon?",
		"We can't guarantee a specific restock date, but you can click 'Notify Me' on the product page and we'll email you the moment it's back.",
		"Is this product discontinued?",
		"product",
	},
	{
		"Does this product come with a warranty?",
		"What warranty do you offer on this item?",
		"All electronics come with a 1-year manufacturer's warranty. Extended warranty plans are available at checkout for 2 or 3 years.",
		"How do I make a warranty claim?",
		"product",
	},
	{
		"How do I make a warranty claim?",
		"My product is faulty — how do I claim the warranty?",
		"Contact us with your order number and a description of the defect. We'll guide you through the warranty claim process and arrange a repair or replacement.",
		"Does this product come with a warranty?",
		"product",
	},
	{
		"Is this product compatible with my device?",
		"Will this item work with my phone?",
		"Compatibility details are listed on the product page under 'Specifications'. If your device isn't listed, contact us and we'll check for you.",
		"Does this product require batteries?",
		"product",
	},
	{
		"Do you have a size guide?",
		"Where can I find sizing information?",
		"Yes! Each clothing and footwear product page has a 'Size Guide' button with detailed measurements. We also offer a live chat fit consultation.",
		"Is this available in my size?",
		"product",
	},
	{
		"Is the product eco-friendly?",
		"Is this item sustainably made?",
		"We're committed to sustainability. Products marked with our 'Eco' badge are made from sustainable materials. Check the product description for specific certifications.",
		"Is this product vegan?",
		"product",
	},
	{
		"How do I care for this product?",
		"What are the care instructions for this item?",
		"Care instructions are printed on the label and listed on the product page under 'Care Guide'. If you can't find them, let us know the product name and we'll share the details.",
		"Does this product come with a warranty?",
		"product",
	},
	{
		"Can I see more photos of the product?",
		"Do you have additional images of this item?",
		"Product pages include multiple photos and videos. You can also check customer reviews for real-world photos. If you need a specific angle, contact us!",
		"Does this product come in different colours?",
		"product",
	},
	{
		"Does this item come in other colours?",
		"Is this product available in different colours?",
		"Colour options are shown as swatches on the product page. Click each swatch to see the item in that colour and check its availability.",
		"Is this product available in my size?",
		"product",
	},

	// ── General inquiries ─────────────────────────────────────────────────────
	{
		"What are your customer support hours?",
		"When is your support team available?",
		"Our support team is available Monday–Friday 8 AM–8 PM and Saturday 9 AM–5 PM (EST). Live chat is available during these hours; email support is 24/7.",
		"How do I contact customer support?",
		"general",
	},
	{
		"How do I contact customer support?",
		"What is the best way to reach your support team?",
		"You can reach us via live chat on our website, email at support@example.com, or by phone at 1-800-555-0100. Our average response time is under 2 minutes.",
		"What are your support hours?",
		"general",
	},
	{
		"Do you have a loyalty rewards program?",
		"Is there a rewards or points program?",
		"Yes! Our Rewards Program lets you earn 1 point per $1 spent. Points can be redeemed for discounts. Sign up for free in Account Settings > Rewards.",
		"How do I earn loyalty points?",
		"general",
	},
	{
		"How do I leave a product review?",
		"Where can I write a review for a product I bought?",
		"Go to 'My Orders', find the item, and click 'Write a Review'. You can rate the product and add photos. Reviews are posted after a brief moderation check.",
		"How do I report a fake review?",
		"general",
	},
	{
		"Do you have a referral program?",
		"Can I earn a reward for referring friends?",
		"Yes! Share your unique referral link from Account Settings > Refer a Friend. You earn $10 credit for each friend who makes their first purchase.",
		"Do you have a loyalty program?",
		"general",
	},
	{
		"Where can I find your privacy policy?",
		"How do you handle my personal data?",
		"Our Privacy Policy is available at the bottom of every page under 'Legal'. It details what data we collect, how it's used, and your rights.",
		"How do I request my data be deleted?",
		"general",
	},
	{
		"How do I report a problem with a seller?",
		"I have an issue with a third-party seller.",
		"To report a seller, go to your order, click 'Report a Problem', and describe the issue. Our Seller Trust team will investigate within 48 hours.",
		"How do I leave a review?",
		"general",
	},
	{
		"Do you offer student discounts?",
		"Is there a discount for students?",
		"Yes! We offer a 15% student discount. Verify your student status through our partner StudentBeans or UNiDAYS and receive a discount code instantly.",
		"Do you offer military discounts?",
		"general",
	},
	{
		"Do you have a physical store?",
		"Are there any brick-and-mortar locations?",
		"We currently operate online only, but we do offer in-store pickup at select partner locations. Check our Store Locator for the nearest pickup point.",
		"Can I return an item in store?",
		"general",
	},
	{
		"How do I use the wishlist feature?",
		"How do I save items for later?",
		"Click the heart icon on any product page to add it to your Wishlist. Access it anytime from 'My Account > Wishlist'. You can share wishlists with friends too.",
		"How do I create a gift list?",
		"general",
	},
}

// ── Multi-turn customer conversations ─────────────────────────────────────────

type convo struct {
	topic        string
	userMsg1     string // customer opening message
	agentMsg1    string // agent first response
	storedMsg2   string // customer follow-up (stored)
	paraphrase2  string // paraphrased follow-up (multiturn_hit query)
	differentMsg string // completely different follow-up (multiturn_miss query)
	answer2      string // agent response to storedMsg2
}

var conversations = []convo{
	{
		"order tracking",
		"Where is my order?",
		"I can see your order is in transit. Let me pull up the tracking details for you.",
		"When exactly will it be delivered?",
		"What is the expected delivery date?",
		"How do I cancel my subscription?", // topic switch → subscription
		"Based on current tracking, your order is expected to arrive this Thursday before 8 PM.",
	},
	{
		"return request",
		"I want to return an item.",
		"Of course! I'd be happy to help you start a return. Could you share your order number?",
		"How do I send the item back?",
		"What is the process for mailing it back to you?",
		"I think my account has been hacked.", // topic switch → account security
		"We'll email you a prepaid return label. Simply pack the item and drop it off at any post office or courier location.",
	},
	{
		"billing dispute",
		"I was charged twice for my order.",
		"I'm so sorry about that! Let me look into this for you right away.",
		"When will the extra charge be refunded?",
		"How long until I get my money back for the duplicate payment?",
		"Do you have this item available in a large size?", // topic switch → product sizing
		"The duplicate charge will automatically be reversed within 3–5 business days. If not, we'll issue a manual refund today.",
	},
	{
		"account login issue",
		"I can't log into my account.",
		"I'm sorry you're having trouble signing in. Let's get this sorted out for you.",
		"I tried resetting my password but I'm not getting the email.",
		"The password reset email hasn't arrived in my inbox.",
		"Do you ship to international addresses?", // topic switch → international shipping
		"Please check your spam folder. If it's not there, verify the email address you registered with — we can update it if needed.",
	},
	{
		"subscription cancellation",
		"I want to cancel my subscription.",
		"I understand. I'm sorry to hear you'd like to cancel. May I ask what's prompting this?",
		"Will I still have access until the end of the billing period?",
		"Can I use the service until my current billing cycle ends?",
		"How do I leave a review for a product I purchased?", // topic switch → product reviews
		"Yes — your access continues until the end of your current billing period. You won't be charged again after today.",
	},
	{
		"damaged item",
		"My item arrived damaged.",
		"I'm so sorry to hear that! This is definitely not the experience we want for you.",
		"Do I need to return the damaged item?",
		"Should I send back the broken product?",
		"Do you have a loyalty rewards program I can join?", // topic switch → loyalty program
		"No need to return it — please keep or dispose of the damaged item. We'll ship a replacement to you today at no charge.",
	},
	{
		"promo code issue",
		"My promo code isn't working.",
		"I'd be happy to look into that for you! Could you share the code you're trying to use?",
		"The code is SAVE20 — it says it's already been used.",
		"I'm trying to use SAVE20 but it shows as already redeemed.",
		"Can I schedule a specific delivery date for my order?", // topic switch → delivery scheduling
		"Codes like SAVE20 are single-use only. Since it's showing as used, it may have been applied to a previous order. I'll issue you a new one-time code now.",
	},
	{
		"wrong item delivered",
		"I received the wrong item.",
		"I sincerely apologize for the mix-up! This shouldn't happen and I'll make it right.",
		"Do I get to keep the wrong item?",
		"Am I able to keep the incorrect product that was sent?",
		"What are your customer support hours?", // topic switch → business hours
		"Absolutely — please keep the item we sent by mistake. We'll ship the correct product to you today with expedited shipping.",
	},
	{
		"shipping delay",
		"My order is delayed.",
		"I understand how frustrating that is. Let me check on the status of your shipment right now.",
		"Is there anything you can do to speed it up?",
		"Can you expedite my delivery?",
		"How do I reset my account password?", // topic switch → account password
		"I've escalated this with the carrier and flagged it as urgent. If it doesn't move within 24 hours, we'll reship your order via overnight delivery at no cost.",
	},
	{
		"payment method update",
		"I need to update my payment method.",
		"Sure! I can guide you through updating your payment information.",
		"Will my existing orders be affected if I change it?",
		"Does changing my payment method impact my pending orders?",
		"Does this product come with a warranty?", // topic switch → warranty
		"Existing orders will use the payment method they were placed with. The new card will apply to future orders and subscription renewals.",
	},
	{
		"refund status",
		"I returned my item two weeks ago but still haven't received my refund.",
		"I sincerely apologize for the delay! Let me check the status of your refund right away.",
		"Can you tell me exactly when it will hit my account?",
		"Do you have a specific date the refund will arrive?",
		"How do I enable two-factor authentication on my account?", // topic switch → account security
		"I can see your refund was processed yesterday. It should appear in your account within 2–3 business days depending on your bank.",
	},
	{
		"product availability",
		"Is this product available in blue?",
		"Let me check the available colours for that item.",
		"Will you be getting more stock of the blue one?",
		"When will the blue version be restocked?",
		"How do I upgrade to the premium subscription plan?", // topic switch → subscription upgrade
		"The blue variant is currently out of stock, but we expect a restock within 2 weeks. I've added you to the notification list!",
	},
	{
		"loyalty points",
		"How do I earn loyalty points?",
		"Great question! Let me explain how our Rewards Program works.",
		"Do I earn points on sale items too?",
		"Can I get points for discounted purchases?",
		"My package shows as delivered but I never received it.", // topic switch → missing delivery
		"Yes! You earn points on all purchases including sale items. The only exception is gift cards and shipping fees.",
	},
	{
		"technical issue checkout",
		"The checkout page keeps freezing.",
		"I'm sorry for the trouble! Let's troubleshoot this together.",
		"I tried a different browser and it's still not working.",
		"Switching browsers didn't fix the checkout issue.",
		"What is your return policy for sale items?", // topic switch → return policy
		"I can place the order for you over chat right now! Please share your cart items and shipping address and I'll process it manually.",
	},
	{
		"size exchange",
		"I ordered the wrong size.",
		"No problem at all! We can arrange an exchange for you.",
		"Do I need to return the wrong size before you send the right one?",
		"Should I send back the incorrect size first before getting the new one?",
		"Do you offer a discount for students?", // topic switch → student discount
		"Yes — please return the incorrect size using the prepaid label we'll email you. Once it's in transit, we'll ship the correct size immediately.",
	},
}

// ── Edge case variants ────────────────────────────────────────────────────────

type edgeVariant struct {
	base    string
	variant string
	answer  string
	kind    string // whitespace | caps | punctuation
}

var edgeVariants = []edgeVariant{
	{"Where is my order?", "  where is my order?  ", "Your order is currently in transit.", "whitespace"},
	{"How do I return an item?", "HOW DO I RETURN AN ITEM?", "Visit My Orders and click Return Item.", "caps"},
	{"My payment was declined.", "my payment was declined", "Please try a different card or payment method.", "punctuation"},
	{"What is your return policy?", "  What is your return policy  ", "Returns are accepted within 30 days.", "whitespace"},
	{"How do I cancel my subscription?", "how do i cancel my subscription", "Go to Account Settings > Subscriptions > Cancel.", "caps"},
	{"I want a refund.", "I WANT A REFUND.", "We'll process your refund within 3–5 business days.", "caps"},
	{"Can I track my order?", "can i track my order?", "Yes, use the tracking link in your shipping confirmation email.", "caps"},
	{"Do you offer free shipping?", "  do you offer free shipping?  ", "Free shipping on orders over $50.", "whitespace"},
	{"How long does delivery take?", "How long does delivery take", "Standard delivery takes 5–7 business days.", "punctuation"},
	{"I need to reset my password.", "  i need to reset my password.  ", "Click Forgot Password on the login page.", "whitespace"},
	{"What payment methods do you accept?", "WHAT PAYMENT METHODS DO YOU ACCEPT?", "We accept Visa, Mastercard, PayPal, and more.", "caps"},
	{"When will my order arrive?", "when will my order arrive", "Check the tracking link in your shipping email.", "caps"},
	{"How do I update my address?", "  How do I update my address?  ", "Go to Account Settings > Addresses.", "whitespace"},
	{"I received a damaged item.", "i received a damaged item.", "Please send us a photo and we'll send a replacement.", "punctuation"},
	{"Can I exchange a product?", "CAN I EXCHANGE A PRODUCT?", "Yes, we offer exchanges within 30 days.", "caps"},
	{"How do I contact support?", "  how do i contact support  ", "Chat, email, or call 1-800-555-0100.", "whitespace"},
	{"Is this item in stock?", "is this item in stock?", "Check the product page for current availability.", "caps"},
	{"How do I apply a coupon?", "How do I apply a coupon", "Enter your code at checkout in the coupon field.", "punctuation"},
	{"I haven't received my refund.", "  i haven't received my refund  ", "Refunds take 3–5 business days after we process them.", "whitespace"},
	{"What is your warranty policy?", "WHAT IS YOUR WARRANTY POLICY?", "All electronics include a 1-year warranty.", "caps"},
	{"How do I create an account?", "how do i create an account?", "Click Sign Up and fill in your details.", "caps"},
	{"My order is missing items.", "  my order is missing items.  ", "Contact us with your order number and we'll send the missing items.", "whitespace"},
	{"Can I pick up in store?", "can i pick up in store?", "Yes, in-store pickup is available at select locations.", "caps"},
	{"I want to upgrade my plan.", "I WANT TO UPGRADE MY PLAN.", "Go to Account Settings > Subscriptions > Change Plan.", "caps"},
	{"How do I delete my account?", "  How do I delete my account?  ", "Go to Account Settings > Privacy > Delete Account.", "whitespace"},
	{"Do you have a mobile app?", "do you have a mobile app?", "Yes! Download it from the App Store or Google Play.", "caps"},
	{"Can I change my order?", "Can I change my order", "Orders can be modified within 30 minutes of purchase.", "punctuation"},
	{"What is your chat support email?", "  What is your chat support email?  ", "Our support email is support@example.com.", "whitespace"},
	{"Do you ship to Canada?", "DO YOU SHIP TO CANADA?", "Yes, we ship to Canada with standard and expedited options.", "caps"},
	{"How do I earn rewards points?", "how do i earn rewards points?", "Earn 1 point per $1 spent on all purchases.", "caps"},
}

// ── Helpers ───────────────────────────────────────────────────────────────────

var counter int

func nextID() string {
	counter++
	return fmt.Sprintf("tc-%04d", counter)
}

func chatReq(text string, temp float64) request {
	return request{
		Type:     "chat",
		Provider: "openai",
		Model:    "gpt-4o",
		Messages: []message{{Role: "user", Content: content{Text: text}}},
		Params:   params{Temperature: temp},
	}
}

func multiTurnReq(msgs []message, temp float64) request {
	return request{
		Type:     "chat",
		Provider: "openai",
		Model:    "gpt-4o",
		Messages: msgs,
		Params:   params{Temperature: temp},
	}
}

func resp(text string) json.RawMessage {
	b, _ := json.Marshal(map[string]string{"content": text})
	return b
}

func basicStored(q, answer string) stored {
	return stored{CacheKey: "eval", Request: chatReq(q, 0.7), Response: resp(answer)}
}

func basicQuery(q string) queryCase {
	return queryCase{CacheKey: "eval", Request: chatReq(q, 0.7)}
}

// ── Generators ────────────────────────────────────────────────────────────────

func genDirectHit(f fact) testCase {
	return testCase{
		ID:          nextID(),
		Category:    "direct_hit",
		Description: fmt.Sprintf("[%s] identical query hits direct O(1) path", f.domain),
		Stored:      basicStored(f.q, f.answer),
		Query:       basicQuery(f.q),
		Expected:    expected{"direct_hit", "Same customer query asked again — must always be a direct hit."},
	}
}

func genSemanticHit(f fact) testCase {
	return testCase{
		ID:          nextID(),
		Category:    "semantic_hit",
		Description: fmt.Sprintf("[%s] rephrased customer query should hit via ANN", f.domain),
		Stored:      basicStored(f.q, f.answer),
		Query:       basicQuery(f.paraphrase),
		Expected:    expected{"semantic_hit", "Same intent rephrased — should exceed similarity threshold."},
	}
}

func genSemanticMiss(f fact) testCase {
	return testCase{
		ID:          nextID(),
		Category:    "semantic_miss",
		Description: fmt.Sprintf("[%s] different intent in same domain must not match", f.domain),
		Stored:      basicStored(f.q, f.answer),
		Query:       basicQuery(f.falseQ),
		Expected:    expected{"miss", "Different customer issue — serving the cached answer would be incorrect."},
	}
}

func genParamIsolation(f fact, temp float64) testCase {
	return testCase{
		ID:          nextID(),
		Category:    "param_isolation",
		Description: fmt.Sprintf("[%s] same query different temperature must not cross buckets", f.domain),
		Stored: stored{
			CacheKey: "eval",
			Request:  chatReq(f.q, 0.7),
			Response: resp(f.answer),
		},
		Query: queryCase{
			CacheKey: "eval",
			Request:  chatReq(f.q, temp),
		},
		Expected: expected{"miss", fmt.Sprintf("Temperature %.1f vs 0.7 → different params_hash bucket.", temp)},
	}
}

func genEdgeCase(v edgeVariant) testCase {
	return testCase{
		ID:          nextID(),
		Category:    "edge_case",
		Description: fmt.Sprintf("[%s] %q normalises to same hash", v.kind, v.base),
		Stored:      basicStored(v.base, v.answer),
		Query:       basicQuery(v.variant),
		Expected:    expected{"direct_hit", fmt.Sprintf("normalizeText() handles %s — same hash as canonical query.", v.kind)},
	}
}

func genConvLimit(q, answer string, nMsgs int) testCase {
	msgs := make([]message, nMsgs)
	for i := range msgs {
		if i%2 == 0 {
			msgs[i] = message{Role: "user", Content: content{Text: q}}
		} else {
			msgs[i] = message{Role: "assistant", Content: content{Text: answer}}
		}
	}
	outcome, note := "skipped", fmt.Sprintf("%d messages > threshold(3) → caching bypassed.", nMsgs)
	if nMsgs <= 3 {
		outcome, note = "direct_hit", fmt.Sprintf("%d messages ≤ threshold(3) → caching proceeds.", nMsgs)
	}
	return testCase{
		ID:          nextID(),
		Category:    "conversation_limit",
		Description: fmt.Sprintf("%d-message support conversation", nMsgs),
		Stored: stored{
			CacheKey: "eval",
			Request:  multiTurnReq(msgs, 0.7),
			Response: resp(answer),
		},
		Query: queryCase{
			CacheKey: "eval",
			Request:  multiTurnReq(msgs, 0.7),
		},
		Expected: expected{outcome, note},
	}
}

func genMultiturnHit(c convo) testCase {
	storedMsgs := []message{
		{Role: "user", Content: content{Text: c.userMsg1}},
		{Role: "assistant", Content: content{Text: c.agentMsg1}},
		{Role: "user", Content: content{Text: c.storedMsg2}},
	}
	queryMsgs := []message{
		{Role: "user", Content: content{Text: c.userMsg1}},
		{Role: "assistant", Content: content{Text: c.agentMsg1}},
		{Role: "user", Content: content{Text: c.paraphrase2}},
	}
	return testCase{
		ID:          nextID(),
		Category:    "multiturn_hit",
		Description: fmt.Sprintf("[%s] paraphrased follow-up in same conversation → semantic hit", c.topic),
		Stored: stored{
			CacheKey: "eval",
			Request:  multiTurnReq(storedMsgs, 0.7),
			Response: resp(c.answer2),
		},
		Query: queryCase{
			CacheKey: "eval",
			Request:  multiTurnReq(queryMsgs, 0.7),
		},
		Expected: expected{"semantic_hit", "Same conversation context + paraphrased follow-up — full embedding should align."},
	}
}

func genMultiturnMiss(c convo) testCase {
	storedMsgs := []message{
		{Role: "user", Content: content{Text: c.userMsg1}},
		{Role: "assistant", Content: content{Text: c.agentMsg1}},
		{Role: "user", Content: content{Text: c.storedMsg2}},
	}
	queryMsgs := []message{
		{Role: "user", Content: content{Text: c.userMsg1}},
		{Role: "assistant", Content: content{Text: c.agentMsg1}},
		{Role: "user", Content: content{Text: c.differentMsg}},
	}
	return testCase{
		ID:          nextID(),
		Category:    "multiturn_miss",
		Description: fmt.Sprintf("[%s] different follow-up intent must not match", c.topic),
		Stored: stored{
			CacheKey: "eval",
			Request:  multiTurnReq(storedMsgs, 0.7),
			Response: resp(c.answer2),
		},
		Query: queryCase{
			CacheKey: "eval",
			Request:  multiTurnReq(queryMsgs, 0.7),
		},
		Expected: expected{"miss", "Same opening but different customer intent — must not serve wrong cached answer."},
	}
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	var cases []testCase

	n := len(facts)
	c := len(conversations)
	e := len(edgeVariants)

	// direct_hit: 120
	for i := 0; i < 120; i++ {
		cases = append(cases, genDirectHit(facts[i%n]))
	}
	// semantic_hit: 240
	for i := 0; i < 240; i++ {
		cases = append(cases, genSemanticHit(facts[i%n]))
	}
	// semantic_miss: 180
	for i := 0; i < 180; i++ {
		cases = append(cases, genSemanticMiss(facts[i%n]))
	}
	// multiturn_hit: 90
	for i := 0; i < 90; i++ {
		cases = append(cases, genMultiturnHit(conversations[i%c]))
	}
	// multiturn_miss: 90
	for i := 0; i < 90; i++ {
		cases = append(cases, genMultiturnMiss(conversations[i%c]))
	}
	// param_isolation: 100
	temps := []float64{0.9, 1.0, 0.5, 0.0}
	for i := 0; i < 100; i++ {
		cases = append(cases, genParamIsolation(facts[i%n], temps[i%len(temps)]))
	}
	// edge_case: 90
	for i := 0; i < 90; i++ {
		cases = append(cases, genEdgeCase(edgeVariants[i%e]))
	}
	// conversation_limit: 90 (alternating 4-msg skipped / 2-msg direct_hit)
	limitQs := []struct{ q, a string }{
		{"Where is my order?", "Your order is in transit and arriving within 2 business days."},
		{"How do I return an item?", "Go to My Orders, select the item, and click Start a Return."},
		{"I want to cancel my subscription.", "Your subscription has been cancelled. Access continues until end of billing period."},
		{"My payment was declined.", "Please check your card details or try a different payment method."},
		{"How long does shipping take?", "Standard shipping takes 5–7 business days."},
		{"I received the wrong item.", "We're sorry! Please keep the wrong item — we'll ship the correct one today."},
		{"How do I reset my password?", "Click Forgot Password on the login page and follow the instructions."},
		{"My package says delivered but I didn't get it.", "Please check with neighbours and contact us if still missing after 24 hours."},
		{"Do you offer free shipping?", "Yes, on all orders over $50."},
		{"How do I contact customer support?", "Chat, email support@example.com, or call 1-800-555-0100."},
	}
	for i := 0; i < 90; i++ {
		lq := limitQs[i%len(limitQs)]
		if i%2 == 0 {
			cases = append(cases, genConvLimit(lq.q, lq.a, 4))
		} else {
			cases = append(cases, genConvLimit(lq.q, lq.a, 2))
		}
	}

	if len(cases) != 1000 {
		fmt.Fprintf(os.Stderr, "ERROR: expected 1000 cases, got %d\n", len(cases))
		os.Exit(1)
	}

	out, err := json.MarshalIndent(cases, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal error: %v\n", err)
		os.Exit(1)
	}

	path := "eval/testcases.json"
	if err := os.WriteFile(path, out, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write error: %v\n", err)
		os.Exit(1)
	}

	counts := make(map[string]int)
	for _, tc := range cases {
		counts[tc.Category]++
	}
	fmt.Printf("Generated %d customer-support test cases → %s\n", len(cases), path)
	for _, cat := range []string{"direct_hit", "semantic_hit", "semantic_miss", "multiturn_hit", "multiturn_miss", "param_isolation", "edge_case", "conversation_limit"} {
		fmt.Printf("  %-22s %d\n", cat, counts[cat])
	}
	fmt.Printf("\nRun: go test ./eval/ -v -run TestSemanticCacheEval -timeout 10m\n")
}
