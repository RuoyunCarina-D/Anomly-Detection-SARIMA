--create channel sub-table
drop table if exists cd_channels;
--3/7/2019
--682,462 rows ~11min, 400s+
CREATE temp table cd_channels as (
select distinct_id,
       property_name,
       first_value(property_value) over (partition by distinct_id, property_name order by time rows between unbounded preceding and unbounded following) as firsttouch_channel,
       last_value(property_value) over (partition by distinct_id, property_name order by time rows between unbounded preceding and unbounded following) as lasttouch_channel
from db.mixpanel_event_properties
where property_name = '[Koch]Publisher'
)
;


--create funnel sub-table
drop table if exists cd_funnel_events;
  --1m 15s --1,114,043 rows
CREATE temp table cd_funnel_events as (
  --de-duplicate for events which should only happen once for one user
  select distinct_id,
         name as event,
         trunc(min(time)) as event_date
  FROM db.mixpanel_events
  where name in ('$ae_first_open', 'TwineAccountCreation')
  group by 1,2

  UNION ALL
  --for events which can exit for multiple times for one user
  select distinct_id,
         name as event,
         trunc(time) as event_date
  FROM db.mixpanel_events
  where name in ('AccountCompletion', 'Goal: active', 'app-install')
)
;


--mapping channel from mixpanel_event_properties to db.singular source
drop table if exists cd_events_channel;
CREATE temp TABLE cd_events_channel as (
	select e.event_date, e.event,
	(case c.firsttouch_channel
	when 'Adbloom - iOS' then 'Adbloom'
	when 'FeedMob - iOS' then 'FeedMob'
	when 'Google Adwords' then 'AdWords'
	when 'Liftoff iOS' then 'Liftoff'
	when 'Twitter-iOS' then 'Twitter'
	when 'Yahoo - Gemini' then 'Yahoo Gemini'
	when 'Oath Ad Platforms' then 'Yahoo Gemini'
	else c.firsttouch_channel
	end) channel,
	c.firsttouch_channel,
  count(*) as values
	from cd_funnel_events e
	left join cd_channels c on e.distinct_id = c.distinct_id
	group by 1,2,3,4
);

--calculate the cost per funnel per channel
CREATE temp TABLE channel_volume_cost as
(select
event_date
,channel
,sum(case when event =  'spend' then values else null end) as spend
,sum(case when event =  'Goal: active' then values else null end) as goal_active
,SUM(case when event =  '$ae_first_open' then values else null end) as first_opens
,SUM(case when event =  'TwineAccountCreation' then values else null end) as TwineAccountCreation
,SUM(case when event =  'app-install' then values else null end) as app_install
,SUM(case when event =  'AccountCompletion' then values else null end) as AccountCompletion
,spend/goal_active as goal_active_cost
,spend/first_opens as first_opens_cost
,spend/TwineAccountCreation as TwineAccountCreation_cost
,spend/app_install as app_install_cost
,spend/AccountCompletion as AccountCompletion_cost
from (
  select *
  from cd_events_channel
  UNION ALL
  select date as event_date,
         'spend' as event,
         source as channel,
         ' ' as firsttouch_channel,
         sum(adn_cost) as values
  from db.singular
  group by 1,2,3,4
)
group by 1,2)

;


drop table if exists ds_prod.cd_channel_cost;
--16s
CREATE TABLE ds_prod.cd_channel_cost as
(select event_date
       ,channel
       ,'goal active' as funnel
       ,goal_active as volumne
       ,goal_active_cost as spend
from channel_volume_cost

union ALL
select event_date
       ,channel
       ,'first opens' as funnel
       ,first_opens as volumne
       ,first_opens_cost as spend
from channel_volume_cost

union ALL
select event_date
       ,channel
       ,'Twine Account Creation' as funnel
       ,TwineAccountCreation as volumne
       ,TwineAccountCreation_cost as spend
from channel_volume_cost

union ALL
select event_date
       ,channel
       ,'App Install' as funnel
       ,app_install as volumne
       ,app_install_cost as spend
from channel_volume_cost

union ALL
select event_date
       ,channel
       ,'Account Completion' as funnel
       ,AccountCompletion as volumne
       ,AccountCompletion_cost as spend
from channel_volume_cost
)
;
