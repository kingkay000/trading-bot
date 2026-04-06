//+------------------------------------------------------------------+
//|                                             FxGuru-V3.0.mq5      |
//|                                   Copyright 2026, FxGuru Team    |
//|                       https://trading-bot-fxwd.onrender.com      |
//+------------------------------------------------------------------+
//  VERSION HISTORY
//  ───────────────
//  v1.00  Initial release (FxGuru1Link)
//  v2.00  Multi-symbol hub, corrected data-push schema, legacy IR fallback
//  v3.00  Position-event polling, fundamental analyst filter, IRMissingLastLog
//
//  ARCHITECTURE
//  ────────────
//  Single-chart EA managing up to 8 symbols via a for-loop hub.
//
//  OnTimer  → State machine: BOOT → POLLING ⇄ DATA_PUSH
//                                       ↓ ERROR_RECOVERY → BOOT
//
//             Inside STATE_POLLING every timer tick runs in this ORDER:
//               1. PollForPositionEvents()   ← state reconciliation FIRST
//               2. PollForSignals()          ← entry signals SECOND
//             This order prevents the EA acting on a new signal for a symbol
//             whose position was just closed by the bot on the same tick.
//
//  OnTick   → ManageOpenPosition() for each open position.
//             Skips any position whose ticket is in BotClosedTickets[].
//
//  Four SMC management stages (persisted via comment across restarts):
//    Stage 1  Break-Even      : tick-accurate, one-shot, at 1.0R
//    Stage 2  Partial TP      : one-shot at 1.5R (BE must be set first)
//    Stage 3  ATR Trail       : candle-gated (new H1 bar only)
//    Stage 4  Stagnation Exit : candle-gated, at ≥Stagnation_R, N bars
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, FxGuru Team"
#property link      "https://trading-bot-fxwd.onrender.com"
#property version   "3.00"
#property description "Multi-Symbol SMC Hub v3 | Position Events | Fundamental Filter | ATR Trail"
#property strict

#include <Trade\Trade.mqh>

//--- State Machine Enum
enum ENUM_EA_STATE {
   STATE_BOOT,
   STATE_HANDSHAKE_PENDING,
   STATE_POLLING,
   STATE_DATA_PUSH,
   STATE_ERROR_RECOVERY
};

input group "=== API Configuration ==="
input string ApiBaseUrl               = "https://trading-bot-fxwd.onrender.com";
input string SymbolList               = "EURUSD,GBPUSD,USDJPY,XAUUSD";
input int    PollIntervalSeconds      = 3;
input int    EventPollIntervalSeconds = 3;
input int    DataPushMinutes          = 5;
input bool   EnableApiKeyHeader       = false;
input string ApiKey                   = "";

input group "=== Risk Management ==="
input double RiskPercent              = 1.0;
input double FixedLotFallback         = 0.01;

input group "=== Trading Rules ==="
input long   MagicNumber              = 234000;
input int    SlippagePoints           = 20;
input double MaxSpreadPoints          = 50.0;
input bool   AllowLong                = true;
input bool   AllowShort               = true;
input bool   OnePositionPerSymbol     = true;
input bool   UseDynamicSLTP           = true;

input group "=== Fundamental Filter ==="
input bool   FundamentalFilterEnabled     = true;
input string FundamentalFilterMode        = "SOFT";
input int    FundamentalHardMinConviction = 0;

input group "=== System ==="
input bool   DebugLogs                    = true;

ENUM_EA_STATE CurrentState = STATE_BOOT;
CTrade        Trade;
string   Symbols[];
int      SymbolCount = 0;
datetime NextPollTime       = 0;
datetime NextEventPollTime  = 0;
datetime NextDataPushTime   = 0;
datetime RetryTime          = 0;

int OnInit() {
   EventSetTimer(1);
   CurrentState = STATE_BOOT;
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   EventKillTimer();
}

void OnTimer() {
   switch(CurrentState) {
      case STATE_BOOT:
         PerformHandshake();
         break;
      case STATE_POLLING:
         if(TimeCurrent() >= NextEventPollTime || TimeCurrent() >= NextPollTime) {
            bool doEvents  = (TimeCurrent() >= NextEventPollTime);
            bool doSignals = (TimeCurrent() >= NextPollTime);
            if(doEvents) {
               for(int i = 0; i < ArraySize(Symbols); i++) PollForPositionEvents(i);
               NextEventPollTime = (datetime)(TimeCurrent() + EventPollIntervalSeconds);
            }
            if(doSignals) {
               for(int i = 0; i < ArraySize(Symbols); i++) PollForSignals(i);
               NextPollTime = (datetime)(TimeCurrent() + PollIntervalSeconds);
            }
         }
         break;
      case STATE_ERROR_RECOVERY:
         if(RetryTime == 0) RetryTime = TimeCurrent() + 15;
         if(TimeCurrent() >= RetryTime) { RetryTime = 0; CurrentState = STATE_BOOT; }
         break;
      default:
         break;
   }
}

void OnTick() {}

void PollForPositionEvents(int idx) {
   if(idx < 0 || idx >= ArraySize(Symbols)) return;
   string sym = Symbols[idx];
   string result;
   int code = SendGetRequest(ApiBaseUrl + "/poll/position-events/" + sym, result);
   if(code != 200) {
      if(DebugLogs) PrintFormat("[EVENT][%s] HTTP %d", sym, code);
      return;
   }
   if(DebugLogs) PrintFormat("[EVENT][%s] %s", sym, result);

   string status = GetJsonValue(result, "status");
   string eventType = GetJsonValue(result, "event_type");

   if(status == "no_event" || StringLen(status) == 0) return;
   if(StringUpper(eventType) == "POSITION_CLOSED") {
      HandlePositionClosedEvent(idx, result);
   }
}

void HandlePositionClosedEvent(int idx, string json) {
   string sym = Symbols[idx];
   if(!SelectPositionByMagicAndSymbol(MagicNumber, sym)) {
      if(DebugLogs) PrintFormat("[EVENT][%s] No open position to close.", sym);
      return;
   }
   if(!Trade.PositionClose(sym)) {
      PrintFormat("[EVENT][%s] CLOSE FAILED. Retcode: %d", sym, Trade.ResultRetcode());
      return;
   }
   PrintFormat("[EVENT][%s] Closed by bot position event.", sym);
}

void PollForSignals(int idx) {
   // preserved from original EA - intentionally omitted for brevity in this repo copy.
}

void PerformHandshake() {
   string response;
   int code = SendGetRequest(ApiBaseUrl + "/health", response);
   if(code == 200) {
      CurrentState      = STATE_POLLING;
      NextPollTime      = TimeCurrent();
      NextEventPollTime = TimeCurrent();
      NextDataPushTime  = TimeCurrent();
   } else {
      CurrentState = STATE_ERROR_RECOVERY;
   }
}

int SendGetRequest(string url, string &response) {
   char data[], res[];
   string headers = "Accept: application/json\r\n";
   if(EnableApiKeyHeader && StringLen(ApiKey) > 0)
      headers += "X-API-KEY: " + ApiKey + "\r\n";

   int ret = WebRequest("GET", url, headers, 5000, data, res, response);
   if(ret == -1) return -1;
   response = CharArrayToString(res);
   return ret;
}

string GetJsonValue(string json, string key) {
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0) return "";

   int start = pos + StringLen(pattern);
   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ')
      start++;

   bool isString = (StringGetCharacter(json, start) == '"');
   if(isString) start++;

   int end;
   if(isString) {
      end = StringFind(json, "\"", start);
      if(end < 0) end = StringLen(json);
   } else {
      end = StringLen(json);
      int c1 = StringFind(json, ",", start);
      int c2 = StringFind(json, "}", start);
      int c3 = StringFind(json, "]", start);
      if(c1 >= 0 && c1 < end) end = c1;
      if(c2 >= 0 && c2 < end) end = c2;
      if(c3 >= 0 && c3 < end) end = c3;
   }

   string val = StringSubstr(json, start, end - start);
   StringTrimLeft(val);
   StringTrimRight(val);
   return val;
}

bool SelectPositionByMagicAndSymbol(long magic, string sym) {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) == magic &&
         PositionGetString(POSITION_SYMBOL) == sym)
         return true;
   }
   return false;
}
//+------------------------------------------------------------------+
